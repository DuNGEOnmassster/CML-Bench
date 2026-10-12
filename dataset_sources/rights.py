"""Validate a curator's asset-specific rights record; never infer legal clearance.

This checks completeness/consistency of supplied evidence, not its legal truth.
Metadata licenses, US-only PD assertions and missing grants cannot pass.
"""

from __future__ import annotations

import re
from datetime import date

LICENSES = {
    "CC0-1.0",
    "CC-BY-4.0",
    "CC-BY-SA-4.0",
    "LicenseRef-Public-Domain-Reviewed",
    "LicenseRef-Worldwide-Grant",
}
PERMISSIONS = ("redistribution", "derivatives", "commercial_use", "training", "evaluation")


def rights_errors(record, source, *, allow_fixtures=False):
    errors = []
    if source["route"] in {"metadata", "excluded"}:
        errors.append("source_not_eligible_for_text")
    if record.get("synthetic_fixture") and not allow_fixtures:
        errors.append("synthetic_fixture_not_production")
    rights = record.get("rights", {})
    if not isinstance(rights, dict):
        return errors + ["rights_not_object"]
    for field in ("reviewed_by", "reviewed_on", "attribution", "access_basis", "assessment"):
        if not isinstance(rights.get(field), str) or not rights[field].strip():
            errors.append("missing_rights_" + field)
    try:
        date.fromisoformat(rights.get("reviewed_on", ""))
    except (ValueError, TypeError):
        errors.append("invalid_review_date")
    if rights.get("status") != "approved" or rights.get("territory") != "worldwide":
        errors.append("worldwide_review_required")
    if not re.fullmatch(r"[a-f0-9]{64}", str(rights.get("asset_sha256", ""))):
        errors.append("asset_hash_required")
    layers = rights.get("layers", [])
    if not isinstance(layers, list) or not all(isinstance(x, dict) for x in layers):
        return errors + ["invalid_rights_layers"]
    required = {"original", "edition"}
    if record.get("is_translation") is True:
        required.add("translation")
    elif record.get("is_translation") is not False:
        errors.append("translation_status_required")
    names = [layer.get("scope") for layer in layers]
    if not all(isinstance(name, str) for name in names):
        return errors + ["invalid_rights_layer_scope"]
    if set(names) != required or len(names) != len(set(names)):
        errors.append("rights_layers_incomplete_or_duplicate")
    for layer in layers:
        label = str(layer.get("scope", "unknown"))
        if not isinstance(layer.get("license"), str) or layer["license"] not in LICENSES:
            errors.append(label + ":license_not_open_track")
        if layer.get("territory") != "worldwide":
            errors.append(label + ":territory_not_worldwide")
        if layer.get("authority_verified") is not True:
            errors.append(label + ":licensor_or_pd_authority_unverified")
        evidence = layer.get("evidence_urls")
        if (
            not isinstance(evidence, list)
            or not evidence
            or not all(isinstance(u, str) and u.startswith("https://") for u in evidence)
        ):
            errors.append(label + ":evidence_required")
        permissions = layer.get("permissions", {})
        if not isinstance(permissions, dict) or any(
            permissions.get(p) is not True for p in PERMISSIONS
        ):
            errors.append(label + ":permissions_incomplete")
        if not isinstance(layer.get("basis"), str) or not layer["basis"].strip():
            errors.append(label + ":basis_required")
    if source["route"] == "permission" and rights.get("separate_grant_verified") is not True:
        errors.append("separate_rightsholder_and_access_grant_required")
    return errors


def license_partition(record):
    licenses = sorted({x["license"] for x in record["rights"]["layers"]})
    # Keep SA material separate; never label it CC0 or assume a single license.
    return "sharealike" if "CC-BY-SA-4.0" in licenses else "open"
