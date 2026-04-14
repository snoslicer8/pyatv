"""Unit tests for pyatv.protocols.companion."""

import asyncio
from ipaddress import ip_address
from unittest.mock import MagicMock

from deepdiff import DeepDiff
import pytest

from pyatv import exceptions
from pyatv.const import DeviceModel, PairingRequirement, Protocol
from pyatv.core import MutableService, mdns
from pyatv.interface import DeviceInfo
from pyatv.protocols.companion import device_info, scan, service_info
from pyatv.protocols.companion.connection import FrameType
from pyatv.protocols.companion.protocol import CompanionProtocol

COMPANION_SERVICE = "_companion-link._tcp.local"


def test_companion_scan_handlers_present():
    handlers = scan()
    assert len(handlers) == 1
    assert COMPANION_SERVICE in handlers


def test_companion_handler_to_service():
    handler, _ = scan()[COMPANION_SERVICE]

    mdns_service = mdns.Service(
        COMPANION_SERVICE, "foo", ip_address("127.0.0.1"), 1234, {"foo": "bar"}
    )
    mdns_response = mdns.Response([], False, None)

    name, service = handler(mdns_service, mdns_response)
    assert name == "foo"
    assert service.port == 1234
    assert service.credentials is None
    assert not DeepDiff(service.properties, {"foo": "bar"})


def test_companion_device_info_name():
    _, device_info_name = scan()[COMPANION_SERVICE]
    assert device_info_name("Ohana") == "Ohana"


@pytest.mark.parametrize(
    "service_type,properties,expected",
    [
        ("_dummy._tcp.local", {"rpmd": "unknown"}, {DeviceInfo.RAW_MODEL: "unknown"}),
        (
            "_dummy._tcp.local",
            {"rpmd": "AppleTV6,2"},
            {DeviceInfo.MODEL: DeviceModel.Gen4K, DeviceInfo.RAW_MODEL: "AppleTV6,2"},
        ),
    ],
)
def test_device_info(service_type, properties, expected):
    assert not DeepDiff(device_info(service_type, properties), expected)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "properties,expected",
    [
        ({}, PairingRequirement.Unsupported),
        ({"rpfl": "0x627B6"}, PairingRequirement.Disabled),
        ({"rpfl": "0x36782"}, PairingRequirement.Mandatory),
        ({"rpfl": "0x0"}, PairingRequirement.Unsupported),
    ],
)
async def test_service_info_pairing(properties, expected):
    service = MutableService(None, Protocol.Companion, 0, properties)

    assert service.pairing == PairingRequirement.Unsupported
    await service_info(service, DeviceInfo({}), {Protocol.Companion: service})
    assert service.pairing == expected


# --- CompanionProtocol queue leak regression ---


def _make_protocol() -> CompanionProtocol:
    """Create a CompanionProtocol with a mocked connection (no real TCP)."""
    connection = MagicMock()
    connection.send = MagicMock()
    srp = MagicMock()
    service = MagicMock()
    service.credentials = None
    protocol = CompanionProtocol(connection, srp, service)
    return protocol


@pytest.mark.asyncio
async def test_exchange_opack_queue_cleaned_up_on_timeout():
    """A timed-out exchange_opack must not leave a stale entry in _queues."""
    protocol = _make_protocol()
    # Record the XID that will be used for this request
    xid_before = protocol._xid

    with pytest.raises((asyncio.TimeoutError, exceptions.ProtocolError)):
        await protocol.exchange_opack(
            FrameType.E_OPACK,
            {"_i": "_ghost", "_t": 2, "_c": {}},
            timeout=0.05,
        )

    assert xid_before not in protocol._queues, (
        "Timed-out XID must be removed from _queues to prevent leak"
    )


@pytest.mark.asyncio
async def test_session_start_request_sends_response():
    """SessionStartRequest from the device must be echoed back as SessionStartResponse."""
    protocol = _make_protocol()

    nonce = b"\x01\x02\x03\x04"
    protocol._handle_session_start_request(nonce)

    protocol.connection.send.assert_called_once_with(FrameType.SessionStartResponse, nonce)


@pytest.mark.asyncio
async def test_session_ready_event_set_by_default():
    """session_ready_event is pre-set so pre-tvOS-26 devices are unaffected."""
    protocol = _make_protocol()
    assert protocol.session_ready_event.is_set()


@pytest.mark.asyncio
async def test_session_ready_event_cleared_and_reset_during_handshake():
    """session_ready_event is cleared during the handshake and set again after."""
    protocol = _make_protocol()

    cleared_during = None

    original_send = protocol.connection.send

    def capturing_send(frame_type, data):
        nonlocal cleared_during
        cleared_during = not protocol.session_ready_event.is_set()
        return original_send(frame_type, data)

    protocol.connection.send = capturing_send
    protocol._handle_session_start_request(b"\xAB\xCD")

    assert cleared_during is True, "Event should be cleared while sending SessionStartResponse"
    assert protocol.session_ready_event.is_set(), "Event should be set again after handshake"
