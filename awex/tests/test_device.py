import pytest

from awex.util import device as device_util


@pytest.mark.parametrize(
    "current_device,visible_devices,expected",
    [
        (8, "", "8"),
        (2, "4,5,6,7", "6"),
        (0, "12", "12"),
        (1, "GPU-a,GPU-b", "GPU-b"),
        (8, "0,1,2,3,4,5,6,7,8,9", "8"),
    ],
)
def test_current_physical_device_id(
    monkeypatch, current_device, visible_devices, expected
):
    monkeypatch.setattr(device_util, "current_device", lambda: current_device)
    monkeypatch.setattr(
        device_util, "visible_devices_env_value", lambda: visible_devices
    )

    assert device_util.current_physical_device_id() == expected


def test_current_physical_device_id_rejects_inconsistent_mapping(monkeypatch):
    monkeypatch.setattr(device_util, "current_device", lambda: 8)
    monkeypatch.setattr(device_util, "visible_devices_env_value", lambda: "4,5,6,7")

    with pytest.raises(RuntimeError, match="cannot be resolved"):
        device_util.current_physical_device_id()
