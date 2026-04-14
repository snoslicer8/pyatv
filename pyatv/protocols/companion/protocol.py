"""Implementation of the Companion protocol."""

import asyncio
from abc import ABC
from enum import Enum
import logging
from random import randint
from typing import Any, Dict, Union

from pyatv import exceptions
from pyatv.auth.hap_pairing import parse_credentials
from pyatv.auth.hap_srp import SRPAuthHandler
from pyatv.interface import BaseService
from pyatv.protocols.companion.auth import CompanionPairVerifyProcedure
from pyatv.protocols.companion.connection import (
    CompanionConnection,
    CompanionConnectionListener,
    FrameType,
)
from pyatv.support import error_handler, log_binary, opack
from pyatv.support.collections import SharedData
from pyatv.support.state_producer import StateProducer

_LOGGER = logging.getLogger(__name__)

_AUTH_FRAMES = [
    FrameType.PS_Start,
    FrameType.PS_Next,
    FrameType.PV_Start,
    FrameType.PV_Next,
]

_OPACK_FRAMES = [
    FrameType.U_OPACK,
    FrameType.E_OPACK,
    FrameType.P_OPACK,
]

DEFAULT_TIMEOUT = 5.0  # Seconds

SRP_SALT = ""
SRP_OUTPUT_INFO = "ClientEncrypt-main"
SRP_INPUT_INFO = "ServerEncrypt-main"

# Either an XID (int) or the frame type (FrameType) is used as an identifier when
# dispatching responses depending on what is supported by a frame. Authentication
# frames never have an XID as multiple authentication attempts cannot be made in
# parallel. Regular OPACK message are however asynchronous and can arrive in any
# order.
FrameIdType = Union[int, FrameType]

# pylint: disable=invalid-name


class MessageType(Enum):
    """Type of message."""

    Event = 1
    Request = 2
    Response = 3


# pylint: enable=invalid-name


class CompanionProtocolListener(ABC):
    """Listener interface for Companion protocol."""

    def event_received(self, event_name: str, data: Dict[str, Any]) -> None:
        """Event was received."""


class CompanionProtocol(
    StateProducer[CompanionProtocolListener], CompanionConnectionListener
):
    """Protocol logic related to Companion."""

    def __init__(
        self,
        connection: CompanionConnection,
        srp: SRPAuthHandler,
        service: BaseService,
    ):
        """Initialize a new CompanionProtocol."""
        super().__init__()
        self.connection = connection
        self.connection.set_listener(self)
        self.srp = srp
        self.service = service
        self._xid: int = randint(0, 2**16)  # Don't know range here, just use something
        self._queues: Dict[FrameIdType, SharedData[Any]] = {}
        self._chacha = None
        self._is_started = False
        # Pre-set so devices that never send SessionStartRequest are unaffected.
        # Cleared transiently while a SessionStartRequest is being answered.
        self._session_ready: asyncio.Event = asyncio.Event()
        self._session_ready.set()

    async def start(self):
        """Connect to device and listen to incoming messages."""
        if self._is_started:
            raise exceptions.ProtocolError("Already started")

        self._is_started = True
        await self.connection.connect()

        if self.service.credentials:
            self.srp.pairing_id = parse_credentials(self.service.credentials).client_id

        _LOGGER.debug("Companion credentials: %s", self.service.credentials)

        await error_handler(self._setup_encryption, exceptions.AuthenticationError)

    @property
    def session_ready_event(self) -> asyncio.Event:
        """Event that is set when no pending SessionStartRequest handshake is in progress."""
        return self._session_ready

    def stop(self):
        """Disconnect from device."""
        self._queues = {}
        self.connection.close()

    async def _setup_encryption(self):
        if self.service.credentials:
            credentials = parse_credentials(self.service.credentials)
            pair_verifier = CompanionPairVerifyProcedure(self, self.srp, credentials)

            await pair_verifier.verify_credentials()
            output_key, input_key = pair_verifier.encryption_keys(
                SRP_SALT, SRP_OUTPUT_INFO, SRP_INPUT_INFO
            )
            self.connection.enable_encryption(output_key, input_key)

    async def exchange_auth(
        self,
        frame_type: FrameType,
        data: Dict[str, Any],
        timeout: float = DEFAULT_TIMEOUT,
    ) -> Dict[str, object]:
        """Exchange an auth frame (PS_* or PV_*)."""
        # Authentication frames have strange logic as *_Start is only used for first
        # message, then *_Next is used for remaining message (even response to first
        # message)
        if frame_type == FrameType.PS_Start:
            identifier = FrameType.PS_Next
        elif frame_type == FrameType.PV_Start:
            identifier = FrameType.PV_Next
        else:
            identifier = frame_type
        return await self._exchange_generic_opack(frame_type, data, identifier, timeout)

    async def exchange_opack(
        self,
        frame_type: FrameType,
        data: Dict[str, Any],
        timeout: float = DEFAULT_TIMEOUT,
    ) -> Dict[str, object]:
        """Send data as OPACK and decode result as OPACK."""
        data["_x"] = self._xid
        identifier = self._xid
        self._xid += 1
        return await self._exchange_generic_opack(frame_type, data, identifier, timeout)

    async def _exchange_generic_opack(
        self,
        frame_type: FrameType,
        data: Dict[str, Any],
        identifier: FrameIdType,
        timeout: float,
    ) -> Dict[str, object]:
        _LOGGER.debug("Exchange OPACK: %s", data)

        self.send_opack(frame_type, data)
        shared_data: SharedData[Any] = SharedData()
        self._queues[identifier] = shared_data
        try:
            unpacked_object = await shared_data.wait(timeout)
        except asyncio.TimeoutError:
            self._queues.pop(identifier, None)
            raise

        if not isinstance(unpacked_object, dict):
            raise exceptions.ProtocolError(
                f"Received unexpected type: {type(unpacked_object)}"
            )

        if "_em" in unpacked_object:
            raise exceptions.ProtocolError(f"Command failed: {unpacked_object['_em']}")

        return unpacked_object

    def send_opack(self, frame_type: FrameType, data: Dict[str, Any]) -> None:
        """Send data encoded with OPACK."""
        # Add XID if not present
        if "_x" not in data:
            data["_x"] = self._xid
            self._xid += 1

        _LOGGER.debug("Send OPACK: %s", data)
        self.connection.send(frame_type, opack.pack(data))

    def frame_received(self, frame_type: FrameType, data: bytes) -> None:
        """Frame was received from remote device."""
        _LOGGER.debug("Received frame %s: %s", frame_type, data)

        if frame_type in _OPACK_FRAMES or frame_type in _AUTH_FRAMES:
            try:
                opack_data, _ = opack.unpack(data)

                if not isinstance(opack_data, dict):
                    _LOGGER.debug("Unsupported OPACK base type: %s", type(opack_data))
                    return

                if frame_type in _AUTH_FRAMES:
                    self._handle_auth(frame_type, opack_data)
                else:
                    self._handle_opack(frame_type, opack_data)
            except Exception:
                _LOGGER.exception("failed to process frame")
        elif frame_type == FrameType.SessionStartRequest:
            self._handle_session_start_request(data)
        else:
            _LOGGER.warning(
                "Received unhandled frame type %s (%d bytes); ignoring",
                frame_type,
                len(data),
            )
            log_binary(_LOGGER, "Unhandled frame payload", Data=data)

    def _handle_auth(self, frame_type: FrameType, opack_data: Dict[str, Any]) -> None:
        _LOGGER.debug("Process incoming auth frame (%s): %s", frame_type, opack_data)
        try:
            shared_data = self._queues.pop(frame_type)
            shared_data.set(opack_data)
        except KeyError:
            _LOGGER.warning("No receiver for auth frame %s", frame_type)

    def _handle_opack(self, frame_type: FrameType, opack_data: Dict[str, Any]) -> None:
        _LOGGER.debug("Process incoming OPACK frame (%s): %s", frame_type, opack_data)

        message_type = opack_data.get("_t")
        if message_type == MessageType.Event.value:
            _LOGGER.debug("Received event: %s", opack_data)
            self.listener.event_received(  # pylint: disable=no-member
                opack_data["_i"], opack_data["_c"]
            )
        elif message_type == MessageType.Response.value:
            xid = opack_data.get("_x")
            if xid in self._queues:
                shared_data = self._queues.pop(xid)
                shared_data.set(opack_data)
            else:
                _LOGGER.debug("No receiver for XID %s", xid)
        else:
            _LOGGER.warning("Got OPACK frame with unsupported type: %s", message_type)

    def _handle_session_start_request(self, data: bytes) -> None:
        """Handle SessionStartRequest sent by tvOS 26+ before it accepts OPACK commands.

        tvOS 26.5 sends this frame (0x10) immediately after the Companion connection is
        established, expecting a SessionStartResponse (0x11) before it will process any
        OPACK commands including FetchAttentionState.

        The exact payload format is not yet fully documented; echoing the received bytes
        is consistent with how other Apple protocol clients handle unknown challenge frames
        and is safe as a starting point until packet captures confirm the full format.
        """
        _LOGGER.debug("Received SessionStartRequest (%d bytes)", len(data))
        log_binary(_LOGGER, "SessionStartRequest payload", Data=data)

        # Signal that we are mid-handshake so that any concurrent exchange_opack calls
        # will wait until after we have sent the response.
        self._session_ready.clear()
        try:
            self.connection.send(FrameType.SessionStartResponse, data)
        finally:
            self._session_ready.set()

        _LOGGER.debug("Sent SessionStartResponse")
