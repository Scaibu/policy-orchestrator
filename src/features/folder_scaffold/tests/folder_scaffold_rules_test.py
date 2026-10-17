from src.features.folder_scaffold.domain.rules.folder_scaffold_rules import (
    can_link,
    infer_layer,
    infer_feature_name,
)


def test_can_link_valid():
    assert can_link("CALLS") is True
    assert can_link("DECLARES_STATE") is True
    assert can_link("CONTROLS_STATE") is True


def test_can_link_invalid():
    assert can_link("RANDOM_INVALID_REL") is False


def test_infer_layer():
    assert infer_layer("Router") == "ingress"
    assert infer_layer("Service") == "domain_core"
    assert infer_layer("UnknownRole") == "general"


def test_infer_feature_name():
    assert infer_feature_name("src/features/wallet/schema/wallet_schema.py") == "wallet"
