"""Тесты Zero-Trust ACL (F-05, F-06).

Каждый негативный сценарий обязан давать отказ: пустые роли пользователя,
пустые ``allowed_roles`` у документа, чужая роль, недоверенный источник
идентичности, подмена пользователя на ``admin``.
"""

from __future__ import annotations

import pytest

from secure_rag.security.acl import (
    AccessControlList,
    AccessDeniedError,
    Principal,
    require_principal,
)


class TestRoleResolution:
    def test_effective_roles_expand_hierarchy(self, acl: AccessControlList) -> None:
        acl.role_hierarchy = {"hr_manager": ["hr_analyst", "hr_viewer"], "hr_analyst": ["hr_viewer"]}
        acl.add_role_to_user("manager", "hr_manager")

        assert acl.get_effective_roles("manager") == {"hr_manager", "hr_analyst", "hr_viewer"}

    def test_hierarchy_cycle_terminates(self, acl: AccessControlList) -> None:
        acl.role_hierarchy = {"a": ["b"], "b": ["a"]}
        acl.add_role_to_user("u", "a")

        assert acl.get_effective_roles("u") == {"a", "b"}

    def test_unknown_user_has_no_roles(self, acl: AccessControlList) -> None:
        assert acl.get_effective_roles("ghost") == set()


class TestChunkAccessDecision:
    @pytest.mark.parametrize(
        ("doc_roles", "user_roles"),
        [
            ([], {"hr"}),  # пустой allowed_roles у документа
            (None, {"hr"}),  # отсутствующее поле
            (["hr"], set()),  # у пользователя нет ролей
            (["hr"], {"engineering"}),  # чужая роль
            (["hr", "legal"], {"legal"}),  # пересечение есть -> доступ
        ],
    )
    def test_decision_matrix(self, acl, doc_roles, user_roles) -> None:
        expected = bool(doc_roles) and bool(user_roles) and bool(set(doc_roles) & user_roles)
        assert acl.chunk_is_accessible(doc_roles, user_roles) is expected

    def test_missing_roles_denied_not_allowed(self, acl: AccessControlList) -> None:
        """Регресс F-05: прежний код пропускал чанк, когда поле отсутствовало."""
        assert acl.chunk_is_accessible([], {"hr"}) is False
        assert acl.chunk_is_accessible(None, {"hr"}) is False


class TestChunkMetadataContract:
    def test_valid_metadata_accepted(self, acl: AccessControlList) -> None:
        assert acl.validate_chunk_metadata({"allowed_roles": ["hr"]}) is True

    def test_missing_field_rejected(self, acl: AccessControlList) -> None:
        assert acl.validate_chunk_metadata({}) is False

    def test_empty_list_rejected(self, acl: AccessControlList) -> None:
        assert acl.validate_chunk_metadata({"allowed_roles": []}) is False

    def test_blank_string_role_rejected(self, acl: AccessControlList) -> None:
        assert acl.validate_chunk_metadata({"allowed_roles": ["  "]}) is False

    def test_json_string_roles_supported(self, acl: AccessControlList) -> None:
        assert acl.validate_chunk_metadata({"allowed_roles": '["hr","legal"]'}) is True

    def test_assert_variant_raises(self, acl: AccessControlList) -> None:
        from secure_rag.security.acl import ACLValidationError

        with pytest.raises(ACLValidationError):
            acl.assert_chunk_metadata({"allowed_roles": []})


class TestWhereFilter:
    def test_filter_uses_user_roles(self, acl: AccessControlList) -> None:
        assert acl.build_where_filter(["hr", "legal"]) == {"allowed_roles": {"$in": ["hr", "legal"]}}

    def test_empty_roles_deny_all(self, acl: AccessControlList) -> None:
        """Пустые роли не должны превращаться в «вернуть всё»."""
        where = acl.build_where_filter([])
        assert where == {"allowed_roles": {"$eq": "__acl_deny_all__"}}


class TestPrincipal:
    def test_roles_normalized(self) -> None:
        principal = Principal(user_id=" u ", roles=[" hr ", "", None, "legal"])
        assert principal.user_id == "u"
        assert principal.roles == ("hr", "legal")

    def test_empty_user_id_rejected(self) -> None:
        with pytest.raises(ValueError):
            Principal(user_id="  ", roles=("hr",))

    def test_principal_without_roles_denied(self) -> None:
        with pytest.raises(AccessDeniedError):
            require_principal(Principal(user_id="u", roles=()))

    def test_untrusted_principal_denied(self) -> None:
        with pytest.raises(AccessDeniedError):
            require_principal(Principal(user_id="u", roles=("hr",), trusted=False))

    def test_none_user_denied(self) -> None:
        with pytest.raises(AccessDeniedError):
            require_principal(None)


class TestAuthorize:
    def test_known_user_authorized(self, acl: AccessControlList) -> None:
        roles = acl.authorize(Principal("hr_user", ("hr",)))
        assert "hr" in roles

    def test_user_without_roles_denied(self, acl: AccessControlList) -> None:
        with pytest.raises(AccessDeniedError):
            acl.authorize(Principal("noroles", ()))

    def test_unknown_user_denied(self, acl: AccessControlList) -> None:
        with pytest.raises(AccessDeniedError):
            acl.authorize(Principal("ghost", ("admin",)))

    def test_impersonation_does_not_grant_roles(self, acl: AccessControlList) -> None:
        """Заявленная роль ``admin`` игнорируется: источник истины — хранилище ACL."""
        roles = acl.authorize(Principal("eng_user", ("admin",)))

        assert "admin" not in roles
        assert roles == {"engineering"}

    def test_unknown_user_with_claimed_admin_denied(self, acl: AccessControlList) -> None:
        """Пользователя нет в хранилище — заявленная роль ``admin`` не помогает."""
        with pytest.raises(AccessDeniedError):
            acl.authorize(Principal("ghost", ("admin",)))


class TestMissingAclStorage:
    def test_absent_storage_denies_everyone(self, settings, tmp_path) -> None:
        """Регресс F-14: исключённый из образа файл не открывает доступ."""
        acl = AccessControlList(settings=settings, storage_path=tmp_path / "missing.json")

        assert acl.known_users() == []
        assert acl.get_effective_roles("admin") == set()
        with pytest.raises(AccessDeniedError):
            acl.authorize(Principal("admin", ("admin",)))


class TestValidateUserAccess:
    @pytest.fixture()
    def acl_with_metadata(self, acl: AccessControlList) -> AccessControlList:
        acl._metadata_cache["doc_hr"] = {"allowed_roles": ["hr"]}
        acl._metadata_cache["doc_eng"] = {"allowed_roles": ["engineering"]}
        acl._metadata_cache["doc_open"] = {"allowed_roles": []}
        acl._metadata_cache["doc_dept"] = {
            "allowed_roles": ["hr"],
            "department": "sales",
        }
        return acl

    def test_user_without_roles_denies_all(self, acl_with_metadata) -> None:
        result = acl_with_metadata.validate_user_access("noroles", ["doc_hr"])

        assert result["allowed"] is False
        assert result["denied_doc_ids"] == ["doc_hr"]
        assert result["reason"] == "User has no roles assigned"

    def test_accessible_and_denied_split(self, acl_with_metadata) -> None:
        result = acl_with_metadata.validate_user_access(
            "hr_user", ["doc_hr", "doc_eng"]
        )

        assert result["allowed"] is True
        assert result["accessible_doc_ids"] == ["doc_hr"]
        assert result["denied_doc_ids"] == ["doc_eng"]

    def test_unknown_metadata_denied(self, acl_with_metadata) -> None:
        result = acl_with_metadata.validate_user_access("hr_user", ["ghost_doc"])

        assert result["denied_doc_ids"] == ["ghost_doc"]

    def test_empty_allowed_roles_denied(self, acl_with_metadata) -> None:
        result = acl_with_metadata.validate_user_access("hr_user", ["doc_open"])

        assert result["denied_doc_ids"] == ["doc_open"]

    def test_department_mismatch_denied(self, acl_with_metadata) -> None:
        acl_with_metadata._user_roles_cache["hr_user"].add("hr")
        # подменяем user_department в кэше через прямое чтение метаданных
        acl_with_metadata._metadata_cache["doc_dept"]["user_department"] = "hr"

        result = acl_with_metadata.validate_user_access("hr_user", ["doc_dept"])

        assert result["denied_doc_ids"] == ["doc_dept"]


class TestAdminOperations:
    def test_roles_of_known_user(self, acl: AccessControlList) -> None:
        assert acl.roles_of("hr_user") == ["hr"]

    def test_roles_of_unknown_user(self, acl: AccessControlList) -> None:
        assert acl.roles_of("ghost") == []

    def test_known_users_sorted(self, acl: AccessControlList) -> None:
        assert acl.known_users() == ["admin", "eng_user", "hr_user", "noroles"]

    def test_remove_role(self, acl: AccessControlList) -> None:
        assert acl.remove_role_from_user("hr_user", "hr") is True
        assert acl.remove_role_from_user("hr_user", "hr") is False
        assert acl.roles_of("hr_user") == []

    def test_clear_role_cache(self, acl: AccessControlList) -> None:
        acl.clear_role_cache()

        assert acl.known_users() == []

