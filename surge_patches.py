"""Surge XT 1.3.4 patch transport and explicitly recorded sanitizer edits."""

import copy
import re
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

JUCE_ALPHABET = ".ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+"
RANDOM_SOURCES = {31: "random_bipolar", 32: "random_unipolar",
                  33: "alternate_bipolar", 34: "alternate_unipolar"}


def decode_memory_block(text: str) -> bytes:
    size_text, encoded = text.split(".", 1)
    size = int(size_text)
    if size < 0 or len(encoded) != (size * 8 + 5) // 6:
        raise ValueError("Invalid JUCE MemoryBlock length")
    result = bytearray(size + 1)
    for index, character in enumerate(encoded):
        value = JUCE_ALPHABET.index(character)
        byte, shift = divmod(index * 6, 8)
        result[byte] |= (value << shift) & 255
        if shift > 2:
            result[byte + 1] |= value >> (8 - shift)
    return bytes(result[:size])


def encode_memory_block(data: bytes) -> str:
    padded = data + b"\0"
    encoded = []
    for offset in range(0, len(data) * 8, 6):
        byte, shift = divmod(offset, 8)
        value = ((padded[byte] | (padded[byte + 1] << 8)) >> shift) & 63
        encoded.append(JUCE_ALPHABET[value])
    return f"{len(data)}." + "".join(encoded)


def component_from_host(state: bytes) -> bytes:
    if state[:4] != b"VC2!" or len(state) < 9:
        raise ValueError("Expected JUCE XML binary state")
    size = struct.unpack_from("<I", state, 4)[0]
    if size > len(state) - 8:
        raise ValueError("Truncated JUCE XML binary state")
    root = ET.fromstring(state[8:8 + size])
    if root.tag != "VST3PluginState":
        raise ValueError(f"Unexpected host state root: {root.tag}")
    component = root.find("IComponent")
    if component is None or component.text is None:
        raise ValueError("Missing IComponent state")
    return decode_memory_block(component.text)


def component_to_host(payload: bytes) -> bytes:
    parse_patch(payload)
    root = ET.Element("VST3PluginState")
    ET.SubElement(root, "IComponent").text = encode_memory_block(payload)
    xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return b"VC2!" + struct.pack("<I", len(xml)) + xml + b"\0"


def read_fxp(path: Path) -> bytes:
    data = path.read_bytes()
    if (len(data) < 60 or data[:4] != b"CcnK" or data[8:12] != b"FPCh"
            or data[16:20] != b"cjs3"):
        raise ValueError(f"Not a Surge FXP: {path}")
    size = struct.unpack_from(">I", data, 56)[0]
    if size != len(data) - 60:
        raise ValueError(f"FXP chunk length mismatch: {path}")
    payload = data[60:]
    parse_patch(payload)
    return payload


def parse_patch(payload: bytes) -> ET.Element:
    if len(payload) < 32 or payload[:4] != b"sub3":
        raise ValueError("Expected Surge sub3 patch container")
    xml_size, *wavetable_sizes = struct.unpack_from("<7I", payload, 4)
    if xml_size == 0 or 32 + xml_size + sum(wavetable_sizes) > len(payload):
        raise ValueError("Truncated patch XML or embedded wavetable")
    root = ET.fromstring(payload[32:32 + xml_size])
    if root.tag != "patch" or root.find("parameters") is None:
        raise ValueError("Expected patch with parameters")
    return root


def replace_patch_xml(payload: bytes, root: ET.Element) -> bytes:
    parse_patch(payload)
    old_size = struct.unpack_from("<I", payload, 4)[0]
    xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return payload[:4] + struct.pack("<I", len(xml)) + payload[8:32] + xml + payload[32 + old_size:]


def audit_patch(payload: bytes) -> dict:
    root = parse_patch(payload)
    revision = int(root.attrib["revision"])
    parameters = root.find("parameters")
    if parameters is None:
        raise ValueError("Missing parameters")
    random_routes = []
    for parameter in parameters:
        for route in parameter.findall("modrouting"):
            source = int(route.attrib["source"])
            if revision < 9 and source > 13:
                source += 1
            if source in RANDOM_SOURCES:
                random_routes.append({
                    "destination": parameter.tag, "source": RANDOM_SOURCES[source],
                    "routing": dict(route.attrib),
                })
    meta = root.find("meta")
    return {
        "revision": revision,
        "meta": {} if meta is None else dict(meta.attrib),
        "random_alternate_routes": random_routes,
        "drift": {p.tag: p.attrib["value"] for p in parameters if p.tag in ("a_drift", "b_drift")},
        "noise_or_formula_lfos": {
            p.tag: int(p.attrib["value"]) for p in parameters
            if re.fullmatch(r"[ab]_lfo\d+_shape", p.tag) and int(p.attrib["value"]) in (4, 5, 9)
        },
        "oscillator_types": {
            p.tag: int(p.attrib["value"]) for p in parameters
            if re.fullmatch(r"[ab]_osc[123]_type", p.tag)
        },
        "noise_mixer_controls": {
            p.tag: dict(p.attrib) for p in parameters if "noise" in p.tag
        },
        "effects": {
            p.tag: dict(p.attrib) for p in parameters if re.fullmatch(r"fx\d+_type", p.tag)
        },
        "limitations": [
            "No external PRNG seed is available through this host.",
            "Noise/formula LFO shapes, noise oscillators, unison and FX randomness are retained.",
            "Retrigger requests the engine's phase reset; zero-degree phase is not guaranteed for every algorithm.",
            "Keytrigger retains the patch's configured start phase.",
            "Freerun advances with DSP/transport state; it need not track wall-clock time.",
        ],
    }


def sanitize_patch(payload: bytes) -> tuple[bytes, dict]:
    root = copy.deepcopy(parse_patch(payload))
    revision = int(root.attrib["revision"])
    if revision < 9:
        raise ValueError("Load and resave legacy patches in Surge before sanitizing")
    parameters = root.find("parameters")
    if parameters is None:
        raise ValueError("Missing parameters")
    by_name = {p.tag: p for p in parameters}
    required = [
        *(f"{scene}_osc{osc}_retrigger" for scene in "ab" for osc in (1, 2, 3)),
        *(f"{scene}_drift" for scene in "ab"),
        *(f"{scene}_lfo{lfo}_trigmode" for scene in "ab" for lfo in range(12)),
    ]
    missing = set(required) - by_name.keys()
    if missing:
        raise ValueError(f"Missing sanitizer controls: {sorted(missing)}")
    edits = []
    removed = []
    for name in required:
        parameter = by_name[name]
        old = parameter.attrib["value"]
        new = "0" if name.endswith("_drift") else "1"
        if float(old) != float(new):
            parameter.set("value", new)
            edits.append({"parameter": name, "before": old, "after": new})
    for parameter in parameters:
        for route in list(parameter.findall("modrouting")):
            source = int(route.attrib["source"])
            if source in RANDOM_SOURCES or parameter.tag in ("a_drift", "b_drift"):
                removed.append({
                    "destination": parameter.tag, "routing": dict(route.attrib),
                    "reason": RANDOM_SOURCES.get(source, "drift_locked_to_zero"),
                })
                parameter.remove(route)
    result = replace_patch_xml(payload, root)
    return result, {
        "policy": "retrigger_on_drift_zero_keytrigger_remove_random_alternate",
        "edits": edits, "removed_routes": removed,
        "before": audit_patch(payload), "after": audit_patch(result),
        "deterministic_claim": False,
    }
