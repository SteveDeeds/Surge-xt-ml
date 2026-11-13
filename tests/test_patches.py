import struct
import xml.etree.ElementTree as ET

import pytest

from surge_patches import (
    audit_patch, component_from_host, component_to_host, decode_memory_block,
    encode_memory_block, parse_patch, read_fxp, sanitize_patch,
)


def patch_fixture() -> bytes:
    root = ET.Element("patch", revision="22")
    ET.SubElement(root, "meta", name="Fixture", author="Test")
    params = ET.SubElement(root, "parameters")
    for scene in "ab":
        for osc in (1, 2, 3):
            ET.SubElement(params, f"{scene}_osc{osc}_retrigger", value="0")
        drift = ET.SubElement(params, f"{scene}_drift", value="0.2")
        ET.SubElement(drift, "modrouting", source="17", depth="0.5")
        for lfo in range(12):
            ET.SubElement(params, f"{scene}_lfo{lfo}_trigmode", value=str(lfo % 3))
    destination = ET.SubElement(params, "a_cutoff", value="1")
    for source in (1, 17, 31, 32, 33, 34):
        ET.SubElement(destination, "modrouting", source=str(source), depth="0.3")
    ET.SubElement(params, "a_lfo0_shape", value="4")
    ET.SubElement(root, "custom_data", preserved="yes")
    xml = ET.tostring(root, encoding="utf-8")
    return b"sub3" + struct.pack("<7I", len(xml), 4, 0, 0, 0, 0, 0) + xml + b"WAVEprivate"


@pytest.mark.parametrize("data", [b"", b"\x00", b"\xff", bytes(range(256)), b"sub3\x00\xff"])
def test_juce_memoryblock_roundtrip(data):
    assert decode_memory_block(encode_memory_block(data)) == data


def test_known_juce_lsb_bit_order():
    assert encode_memory_block(b"\xff") == "1.+C"
    assert encode_memory_block(b"\x01") == "1.A."


def test_host_roundtrip_and_embedded_data_preservation():
    payload = patch_fixture()
    assert component_from_host(component_to_host(payload)) == payload
    result, report = sanitize_patch(payload)
    assert result.endswith(b"WAVEprivate")
    root = parse_patch(result)
    assert root.find("custom_data").get("preserved") == "yes"
    for scene in "ab":
        assert root.find(f"parameters/{scene}_drift").get("value") == "0"
        assert not root.findall(f"parameters/{scene}_drift/modrouting")
        for osc in (1, 2, 3):
            assert root.find(f"parameters/{scene}_osc{osc}_retrigger").get("value") == "1"
        for lfo in range(12):
            assert root.find(f"parameters/{scene}_lfo{lfo}_trigmode").get("value") == "1"
    assert [r.get("source") for r in root.findall("parameters/a_cutoff/modrouting")] == ["1", "17"]
    assert root.find("parameters/a_lfo0_shape").get("value") == "4"
    assert not audit_patch(result)["random_alternate_routes"]
    assert len(report["removed_routes"]) == 6
    assert report["deterministic_claim"] is False
    again, second_report = sanitize_patch(result)
    assert parse_patch(again).find("parameters/a_cutoff").attrib == root.find("parameters/a_cutoff").attrib
    assert second_report["edits"] == second_report["removed_routes"] == []


def test_truncated_patch_and_foreign_fxp_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="Truncated"):
        parse_patch(patch_fixture()[:40])
    path = tmp_path / "foreign.fxp"
    path.write_bytes(b"not Surge")
    with pytest.raises(ValueError, match="Not a Surge"):
        read_fxp(path)
    data = b"CcnK" + b"\0" * 4 + b"FPCh" + b"\0" * 4 + b"cjs3" + b"\0" * 36 + struct.pack(">I", 999)
    path.write_bytes(data)
    with pytest.raises(ValueError, match="length mismatch"):
        read_fxp(path)


def test_legacy_random_source_ids_are_audited_but_not_sanitized():
    payload = patch_fixture()
    root = parse_patch(payload)
    root.set("revision", "8")
    for route in root.iter("modrouting"):
        if int(route.get("source")) > 13:
            route.set("source", str(int(route.get("source")) - 1))
    from surge_patches import replace_patch_xml
    legacy = replace_patch_xml(payload, root)
    assert len(audit_patch(legacy)["random_alternate_routes"]) == 4
    with pytest.raises(ValueError, match="resave legacy"):
        sanitize_patch(legacy)
