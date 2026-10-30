import hashlib
import json
from typing import Any


def build_untrusted_content_record(
    *,
    content_id: str,
    source_type: str,
    content: str,
    source_reference: str,
    resource_tenant_id: str | None,
) -> dict[str, Any]:
    """
    为进入 Agent 的外部内容建立不可信内容记录。

    功能：
        校验内容标识、来源类型、正文、来源引用和资源租户标识，
        再生成包含来源信息、稳定内容摘要和固定安全标签的记录。

        用户输入、RAG 文档、网页内容和工具输出即使包含
        “系统命令”“管理员授权”等文字，也只能作为数据使用，
        不能因为正文中的自我声明而获得指令权或权限。

        content_sha256 用于识别和审计本次读取的具体内容，
        不能证明内容真实、安全或没有被篡改。

        resource_tenant_id 表示数据层随资源返回的租户归属，
        后续仍需与认证系统提供的可信租户标识进行比较。
        本函数不会仅凭该字段批准跨租户访问。

        本函数不检测恶意关键词，不调用 LLM，
        不执行内容中的任何指令，也不修改输入值。

    参数：
        content_id:
            本次外部内容的非空唯一标识。

        source_type:
            内容来源类型，只允许 user_input、rag_document、
            web_content 或 tool_output。

        content:
            外部内容正文，必须是非空字符串。

        source_reference:
            可审计的来源引用，例如文档 ID、网页地址、
            工具调用 ID 或消息 ID，必须是非空字符串。

        resource_tenant_id:
            数据层声明的资源租户标识；公共资源可传 None。
            它是待校验的数据属性，不是当前调用者身份。

    返回值：
        dict[str, Any]:
            不可信内容记录，格式如下：

            {
                "content_id": "rag-001",
                "source_type": "rag_document",
                "content": "文档正文",
                "source_reference": "document:dog-care-001",
                "resource_tenant_id": "tenant-a",
                "trust_level": "untrusted",
                "instruction_authority": False,
                "handling_action": "treat_as_data",
                "content_sha256": "...",
            }

            instruction_authority 固定为 False，表示该内容
            不能覆盖系统策略、修改权限或批准工具动作。

            handling_action 固定为 treat_as_data，表示下游
            可以提取事实，但不能直接执行正文中的指令。

    异常：
        TypeError:
            任一参数类型不符合约定时抛出。

        ValueError:
            字符串为空，或 source_type 不在允许集合时抛出。
    """
    string_parameters = {
        "content_id": content_id,
        "source_type": source_type,
        "content": content,
        "source_reference": source_reference,
    }
    for parameter_name, parameter_value in string_parameters.items():
        if not isinstance(parameter_value, str):
            raise TypeError(f"{parameter_name} 必须是字符串")
        if not parameter_value.strip():
            raise ValueError(f"{parameter_name} 不能为空")

    if resource_tenant_id is not None:
        if not isinstance(resource_tenant_id, str):
            raise TypeError(
                "resource_tenant_id 必须是字符串或 None"
            )
        if not resource_tenant_id.strip():
            raise ValueError(
                "resource_tenant_id 不能是空字符串"
            )

    allowed_source_types = {
        "user_input",
        "rag_document",
        "web_content",
        "tool_output",
    }
    if source_type not in allowed_source_types:
        raise ValueError(
            "source_type 必须是 user_input、rag_document、"
            "web_content 或 tool_output"
        )

    content_sha256 = hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()

    return {
        "content_id": content_id,
        "source_type": source_type,
        "content": content,
        "source_reference": source_reference,
        "resource_tenant_id": resource_tenant_id,
        "trust_level": "untrusted",
        "instruction_authority": False,
        "handling_action": "treat_as_data",
        "content_sha256": content_sha256,
    }


def build_authenticated_security_context(
    *,
    authenticated_user_id: str,
    authenticated_tenant_id: str,
    granted_permissions: list[str],
    authentication_method: str,
    authentication_event_id: str,
) -> dict[str, Any]:
    """
    根据认证层已验证的结果建立可信安全上下文。

    功能：
        校验已认证用户、已认证租户、权限列表、认证方式
        和认证事件标识的结构，再生成后续权限判断所使用的
        Trusted Security Context（可信安全上下文）。

        权限名称会去除首尾空白、去重并按字典序排列，
        便于产生稳定的审计输出和确定性的权限判断。

        本函数不读取用户自然语言、RAG 文档或 LLM 输出，
        也不允许这些不可信内容覆盖已认证身份与权限。

        本函数本身不校验密码、Session、JWT、API Key，
        也不向身份服务查询权限。调用方必须确保所有参数
        来自已经完成验证的认证与授权基础设施；
        不能直接把客户端请求体中的同名字段传入本函数。

    参数：
        authenticated_user_id:
            认证系统确认的非空用户标识。

        authenticated_tenant_id:
            认证系统确认的非空租户标识。

        granted_permissions:
            授权系统授予当前身份的权限名称列表。
            每一项必须是非空字符串；允许传入空列表，
            表示当前身份没有获得任何业务权限。

        authentication_method:
            实际完成认证的方法，例如 session、jwt 或 api_key。
            这里只做记录，不根据名称推断认证强度。

        authentication_event_id:
            本次认证结果对应的非空审计事件标识，
            便于后续权限决定追踪身份来源。

    返回值：
        dict[str, Any]:
            可信安全上下文，格式如下：

            {
                "authenticated_user_id": "user-001",
                "authenticated_tenant_id": "tenant-a",
                "granted_permissions": ["document:read"],
                "authentication_method": "session",
                "authentication_event_id": "auth-event-001",
                "identity_verified": True,
                "security_context_source": "authentication_layer",
            }

            identity_verified 固定为 True，表示这个结构只能在
            上游认证成功后创建，不表示本函数亲自完成了认证。

            security_context_source 固定为 authentication_layer，
            提醒下游这里的数据不能由模型或外部内容替换。

    异常：
        TypeError:
            参数类型不符合约定时抛出。

        ValueError:
            标识、认证方式或权限名称为空时抛出。
    """
    string_parameters = {
        "authenticated_user_id": authenticated_user_id,
        "authenticated_tenant_id": authenticated_tenant_id,
        "authentication_method": authentication_method,
        "authentication_event_id": authentication_event_id,
    }
    for parameter_name, parameter_value in string_parameters.items():
        if not isinstance(parameter_value, str):
            raise TypeError(f"{parameter_name} 必须是字符串")
        if not parameter_value.strip():
            raise ValueError(f"{parameter_name} 不能为空")

    if not isinstance(granted_permissions, list):
        raise TypeError("granted_permissions 必须是列表")

    normalized_permissions: set[str] = set()
    for permission in granted_permissions:
        if not isinstance(permission, str):
            raise TypeError("权限名称必须是字符串")
        normalized_permission = permission.strip()
        if not normalized_permission:
            raise ValueError("权限名称不能为空")
        normalized_permissions.add(normalized_permission)

    return {
        "authenticated_user_id": authenticated_user_id,
        "authenticated_tenant_id": authenticated_tenant_id,
        "granted_permissions": sorted(normalized_permissions),
        "authentication_method": authentication_method,
        "authentication_event_id": authentication_event_id,
        "identity_verified": True,
        "security_context_source": "authentication_layer",
    }


def evaluate_tenant_resource_access(
    *,
    security_context: dict[str, Any],
    content_record: dict[str, Any],
    required_permission: str,
) -> dict[str, Any]:
    """
    根据可信身份、资源租户归属和所需权限评估读取准入。

    功能：
        校验安全上下文确实声明来自认证层，
        并校验外部内容仍保持不可信数据标签。

        对带有 resource_tenant_id 的租户私有资源，
        要求它与认证上下文中的租户完全一致；
        对 resource_tenant_id=None 的公共资源跳过租户比较。

        无论资源是否公共，都要求当前身份拥有
        required_permission。租户一致只表示数据边界一致，
        不代表用户自动拥有读取权限。

        本函数返回内部授权决定，不修改任何输入，
        不执行查询，也不把资源正文交给模型。

        真实系统必须在数据库、向量检索或对象存储查询时，
        就使用可信 tenant_id 限制数据范围。本函数是应用层
        Defense in Depth（纵深防御），不能弥补已经发生的
        越权数据读取。

    参数：
        security_context:
            build_authenticated_security_context 返回的
            可信安全上下文。

        content_record:
            build_untrusted_content_record 返回的外部内容记录。
            resource_tenant_id 用于本次租户边界比较。

        required_permission:
            访问当前资源必须具备的非空权限名称，
            例如 document:read。

    返回值：
        dict[str, Any]:
            内部访问决定，包含：

            allowed:
                是否通过租户和权限检查。

            effective_action:
                allow 或 reject。

            decision_reason:
                allowed、tenant_mismatch 或
                missing_required_permission。

            authenticated_user_id、authenticated_tenant_id、
            resource_tenant_id、required_permission 和
            authentication_event_id：用于内部审计。

            decision_reason 是内部原因，不应直接向外部用户
            暴露资源是否存在或属于哪个租户。外部响应应使用
            统一的无权访问信息，防止资源枚举。

    异常：
        TypeError:
            参数或关键字段类型不符合约定时抛出。

        ValueError:
            缺少字段、安全来源标签被破坏、权限名称为空，
            或关键值不符合约定时抛出。
    """
    if not isinstance(security_context, dict):
        raise TypeError("security_context 必须是字典")
    if not isinstance(content_record, dict):
        raise TypeError("content_record 必须是字典")
    if not isinstance(required_permission, str):
        raise TypeError("required_permission 必须是字符串")
    normalized_required_permission = required_permission.strip()
    if not normalized_required_permission:
        raise ValueError("required_permission 不能为空")

    required_security_fields = {
        "authenticated_user_id",
        "authenticated_tenant_id",
        "granted_permissions",
        "authentication_event_id",
        "identity_verified",
        "security_context_source",
    }
    missing_security_fields = required_security_fields.difference(
        security_context
    )
    if missing_security_fields:
        raise ValueError(
            "security_context 缺少字段："
            f"{', '.join(sorted(missing_security_fields))}"
        )

    security_string_fields = {
        "authenticated_user_id": security_context[
            "authenticated_user_id"
        ],
        "authenticated_tenant_id": security_context[
            "authenticated_tenant_id"
        ],
        "authentication_event_id": security_context[
            "authentication_event_id"
        ],
    }
    for field_name, field_value in security_string_fields.items():
        if not isinstance(field_value, str):
            raise TypeError(f"{field_name} 必须是字符串")
        if not field_value.strip():
            raise ValueError(f"{field_name} 不能为空")

    if security_context["identity_verified"] is not True:
        raise ValueError("security_context 必须来自已验证身份")
    if (
        security_context["security_context_source"]
        != "authentication_layer"
    ):
        raise ValueError("security_context 来源必须是认证层")

    granted_permissions = security_context["granted_permissions"]
    if not isinstance(granted_permissions, list):
        raise TypeError("granted_permissions 必须是列表")
    for permission in granted_permissions:
        if not isinstance(permission, str):
            raise TypeError("权限名称必须是字符串")
        if not permission.strip():
            raise ValueError("权限名称不能为空")

    required_content_fields = {
        "content_id",
        "resource_tenant_id",
        "trust_level",
        "instruction_authority",
        "handling_action",
    }
    missing_content_fields = required_content_fields.difference(
        content_record
    )
    if missing_content_fields:
        raise ValueError(
            "content_record 缺少字段："
            f"{', '.join(sorted(missing_content_fields))}"
        )
    if content_record["trust_level"] != "untrusted":
        raise ValueError("外部内容必须保持 untrusted 标签")
    if content_record["instruction_authority"] is not False:
        raise ValueError("外部内容不能拥有指令权")
    if content_record["handling_action"] != "treat_as_data":
        raise ValueError("外部内容必须按数据处理")

    resource_tenant_id = content_record["resource_tenant_id"]
    if resource_tenant_id is not None:
        if not isinstance(resource_tenant_id, str):
            raise TypeError(
                "resource_tenant_id 必须是字符串或 None"
            )
        if not resource_tenant_id.strip():
            raise ValueError(
                "resource_tenant_id 不能是空字符串"
            )

    authenticated_tenant_id = security_context[
        "authenticated_tenant_id"
    ]
    if (
        resource_tenant_id is not None
        and resource_tenant_id != authenticated_tenant_id
    ):
        allowed = False
        effective_action = "reject"
        decision_reason = "tenant_mismatch"
    elif (
        normalized_required_permission
        not in granted_permissions
    ):
        allowed = False
        effective_action = "reject"
        decision_reason = "missing_required_permission"
    else:
        allowed = True
        effective_action = "allow"
        decision_reason = "allowed"

    return {
        "allowed": allowed,
        "effective_action": effective_action,
        "decision_reason": decision_reason,
        "authenticated_user_id": security_context[
            "authenticated_user_id"
        ],
        "authenticated_tenant_id": authenticated_tenant_id,
        "resource_tenant_id": resource_tenant_id,
        "required_permission": normalized_required_permission,
        "authentication_event_id": security_context[
            "authentication_event_id"
        ],
    }


def evaluate_tool_action_permission(
    *,
    security_context: dict[str, Any],
    tool_policy: dict[str, Any],
    user_confirmation: str,
) -> dict[str, Any]:
    """
    根据服务端工具策略、身份权限和用户确认评估工具准入。

    功能：
        校验安全上下文来自认证层，并校验工具策略声明来自
        server_tool_registry（服务端工具注册表）。

        使用固定顺序评估：工具是否启用、租户是否在允许范围、
        身份是否拥有所需权限、用户是否明确拒绝，
        最后才判断是否需要请求确认或允许执行。

        用户确认不能创造权限。即使 user_confirmation 为
        confirmed，只要工具未启用、租户不允许或权限不足，
        最终仍然必须 reject。

        本函数只产生内部权限决定，不执行工具，
        不校验工具参数，也不持久化确认结果。
        当前学习版还没有把确认绑定到具体参数指纹；
        正式系统必须防止确认后参数被替换。

    参数：
        security_context:
            认证层提供的可信安全上下文。

        tool_policy:
            服务端工具注册表提供的策略，格式如下：

            {
                "tool_name": "send_email",
                "enabled": True,
                "required_permission": "tool:email:send",
                "confirmation_required": True,
                "allowed_tenant_ids": ["tenant-a"],
                "policy_source": "server_tool_registry",
            }

            allowed_tenant_ids=None 表示该工具没有额外的
            租户允许名单限制，但仍需通过权限检查。

        user_confirmation:
            当前具体工具动作的用户确认状态，只允许
            not_provided、confirmed 或 rejected。

    返回值：
        dict[str, Any]:
            工具准入决定，包含：

            allowed:
                当前是否可以进入真实工具执行阶段。

            effective_action:
                allow、require_confirmation 或 reject。

            decision_reason:
                tool_disabled、tenant_not_allowed、
                missing_required_permission、user_rejected、
                confirmation_required 或 allowed。

            tool_name、required_permission、
            authenticated_user_id、authenticated_tenant_id、
            authentication_event_id：用于内部审计。

            内部拒绝原因不应直接暴露工具登记和租户细节。

    异常：
        TypeError:
            参数或策略字段类型不符合约定时抛出。

        ValueError:
            缺字段、来源标记不正确、字符串为空，
            租户名单不合法或确认状态不受支持时抛出。
    """
    if not isinstance(security_context, dict):
        raise TypeError("security_context 必须是字典")
    if not isinstance(tool_policy, dict):
        raise TypeError("tool_policy 必须是字典")
    if not isinstance(user_confirmation, str):
        raise TypeError("user_confirmation 必须是字符串")

    allowed_confirmation_states = {
        "not_provided",
        "confirmed",
        "rejected",
    }
    if user_confirmation not in allowed_confirmation_states:
        raise ValueError(
            "user_confirmation 必须是 not_provided、"
            "confirmed 或 rejected"
        )

    required_security_fields = {
        "authenticated_user_id",
        "authenticated_tenant_id",
        "granted_permissions",
        "authentication_event_id",
        "identity_verified",
        "security_context_source",
    }
    missing_security_fields = required_security_fields.difference(
        security_context
    )
    if missing_security_fields:
        raise ValueError(
            "security_context 缺少字段："
            f"{', '.join(sorted(missing_security_fields))}"
        )
    if security_context["identity_verified"] is not True:
        raise ValueError("security_context 必须来自已验证身份")
    if (
        security_context["security_context_source"]
        != "authentication_layer"
    ):
        raise ValueError("security_context 来源必须是认证层")

    security_string_fields = {
        "authenticated_user_id": security_context[
            "authenticated_user_id"
        ],
        "authenticated_tenant_id": security_context[
            "authenticated_tenant_id"
        ],
        "authentication_event_id": security_context[
            "authentication_event_id"
        ],
    }
    for field_name, field_value in security_string_fields.items():
        if not isinstance(field_value, str):
            raise TypeError(f"{field_name} 必须是字符串")
        if not field_value.strip():
            raise ValueError(f"{field_name} 不能为空")

    granted_permissions = security_context["granted_permissions"]
    if not isinstance(granted_permissions, list):
        raise TypeError("granted_permissions 必须是列表")
    for permission in granted_permissions:
        if not isinstance(permission, str):
            raise TypeError("权限名称必须是字符串")
        if not permission.strip():
            raise ValueError("权限名称不能为空")

    required_policy_fields = {
        "tool_name",
        "enabled",
        "required_permission",
        "confirmation_required",
        "allowed_tenant_ids",
        "policy_source",
    }
    missing_policy_fields = required_policy_fields.difference(
        tool_policy
    )
    if missing_policy_fields:
        raise ValueError(
            "tool_policy 缺少字段："
            f"{', '.join(sorted(missing_policy_fields))}"
        )

    policy_string_fields = {
        "tool_name": tool_policy["tool_name"],
        "required_permission": tool_policy["required_permission"],
    }
    for field_name, field_value in policy_string_fields.items():
        if not isinstance(field_value, str):
            raise TypeError(f"{field_name} 必须是字符串")
        if not field_value.strip():
            raise ValueError(f"{field_name} 不能为空")
    if type(tool_policy["enabled"]) is not bool:
        raise TypeError("enabled 必须是布尔值")
    if type(tool_policy["confirmation_required"]) is not bool:
        raise TypeError("confirmation_required 必须是布尔值")
    if tool_policy["policy_source"] != "server_tool_registry":
        raise ValueError("tool_policy 必须来自服务端工具注册表")

    allowed_tenant_ids = tool_policy["allowed_tenant_ids"]
    if allowed_tenant_ids is not None:
        if not isinstance(allowed_tenant_ids, list):
            raise TypeError(
                "allowed_tenant_ids 必须是列表或 None"
            )
        for tenant_id in allowed_tenant_ids:
            if not isinstance(tenant_id, str):
                raise TypeError("允许的租户标识必须是字符串")
            if not tenant_id.strip():
                raise ValueError("允许的租户标识不能为空")

    authenticated_tenant_id = security_context[
        "authenticated_tenant_id"
    ]
    required_permission = tool_policy["required_permission"].strip()
    if not tool_policy["enabled"]:
        allowed = False
        effective_action = "reject"
        decision_reason = "tool_disabled"
    elif (
        allowed_tenant_ids is not None
        and authenticated_tenant_id not in allowed_tenant_ids
    ):
        allowed = False
        effective_action = "reject"
        decision_reason = "tenant_not_allowed"
    elif required_permission not in granted_permissions:
        allowed = False
        effective_action = "reject"
        decision_reason = "missing_required_permission"
    elif user_confirmation == "rejected":
        allowed = False
        effective_action = "reject"
        decision_reason = "user_rejected"
    elif (
        tool_policy["confirmation_required"]
        and user_confirmation != "confirmed"
    ):
        allowed = False
        effective_action = "require_confirmation"
        decision_reason = "confirmation_required"
    else:
        allowed = True
        effective_action = "allow"
        decision_reason = "allowed"

    return {
        "allowed": allowed,
        "effective_action": effective_action,
        "decision_reason": decision_reason,
        "tool_name": tool_policy["tool_name"],
        "required_permission": required_permission,
        "authenticated_user_id": security_context[
            "authenticated_user_id"
        ],
        "authenticated_tenant_id": authenticated_tenant_id,
        "authentication_event_id": security_context[
            "authentication_event_id"
        ],
    }


def build_tool_confirmation_record(
    *,
    security_context: dict[str, Any],
    permission_decision: dict[str, Any],
    tool_arguments: dict[str, Any],
    confirmation_id: str,
    user_decision: str,
) -> dict[str, Any]:
    """
    将用户确认绑定到具体工具、参数、身份和租户。

    功能：
        只接受权限门已经返回 require_confirmation 的决定，
        对工具参数生成 Canonical JSON（规范 JSON）指纹，
        再记录本次用户是 confirmed 还是 rejected。

        后续执行前应使用同样规则重新计算工具名和参数指纹，
        并比较用户、租户、认证事件与确认记录。
        任一项变化都不能复用这次确认。

        本函数避免把确认建模成全局 confirmed=True，
        但不亲自展示确认界面，也不验证用户交互通道。
        调用方必须是受信的确认恢复处理层，不能直接相信
        LLM、RAG 文档或普通请求正文提供的 user_decision。

        本函数不重新执行权限检查、不调用工具，
        也不保存原始工具参数；参数摘要用于绑定和审计，
        不能还原参数，也不能证明参数本身安全。

    参数：
        security_context:
            当前认证层提供的可信安全上下文。

        permission_decision:
            evaluate_tool_action_permission 返回的内部决定。
            effective_action 必须为 require_confirmation，
            decision_reason 必须为 confirmation_required。

        tool_arguments:
            等待用户确认的完整工具参数字典。
            必须能够安全序列化为规范 JSON。

        confirmation_id:
            本次确认请求的非空唯一标识。

        user_decision:
            用户通过受信交互通道作出的决定，
            只能是 confirmed 或 rejected。

    返回值：
        dict[str, Any]:
            绑定后的确认记录，包含：

            confirmation_id、tool_name、arguments_sha256、
            user_decision、confirmed、confirmed_by_user_id、
            authenticated_tenant_id、authentication_event_id
            和 record_source。

            record_source 固定为 trusted_confirmation_handler，
            表示记录应由受信确认处理层产生；
            它不是防伪签名，也不能代替受保护的数据流。

    异常：
        TypeError:
            参数或关键字段类型不符合约定时抛出。

        ValueError:
            标识为空、身份来源不可信、权限决定不要求确认、
            用户决定不受支持，或参数无法转换为规范 JSON 时抛出。
    """
    if not isinstance(security_context, dict):
        raise TypeError("security_context 必须是字典")
    if not isinstance(permission_decision, dict):
        raise TypeError("permission_decision 必须是字典")
    if not isinstance(tool_arguments, dict):
        raise TypeError("tool_arguments 必须是字典")
    if not isinstance(confirmation_id, str):
        raise TypeError("confirmation_id 必须是字符串")
    if not confirmation_id.strip():
        raise ValueError("confirmation_id 不能为空")
    if not isinstance(user_decision, str):
        raise TypeError("user_decision 必须是字符串")
    if user_decision not in {"confirmed", "rejected"}:
        raise ValueError(
            "user_decision 必须是 confirmed 或 rejected"
        )

    required_security_fields = {
        "authenticated_user_id",
        "authenticated_tenant_id",
        "authentication_event_id",
        "identity_verified",
        "security_context_source",
    }
    missing_security_fields = required_security_fields.difference(
        security_context
    )
    if missing_security_fields:
        raise ValueError(
            "security_context 缺少字段："
            f"{', '.join(sorted(missing_security_fields))}"
        )
    if security_context["identity_verified"] is not True:
        raise ValueError("security_context 必须来自已验证身份")
    if (
        security_context["security_context_source"]
        != "authentication_layer"
    ):
        raise ValueError("security_context 来源必须是认证层")

    security_string_fields = {
        "authenticated_user_id": security_context[
            "authenticated_user_id"
        ],
        "authenticated_tenant_id": security_context[
            "authenticated_tenant_id"
        ],
        "authentication_event_id": security_context[
            "authentication_event_id"
        ],
    }
    for field_name, field_value in security_string_fields.items():
        if not isinstance(field_value, str):
            raise TypeError(f"{field_name} 必须是字符串")
        if not field_value.strip():
            raise ValueError(f"{field_name} 不能为空")

    required_decision_fields = {
        "allowed",
        "effective_action",
        "decision_reason",
        "tool_name",
        "authenticated_user_id",
        "authenticated_tenant_id",
        "authentication_event_id",
    }
    missing_decision_fields = required_decision_fields.difference(
        permission_decision
    )
    if missing_decision_fields:
        raise ValueError(
            "permission_decision 缺少字段："
            f"{', '.join(sorted(missing_decision_fields))}"
        )
    if permission_decision["allowed"] is not False:
        raise ValueError("待确认决定的 allowed 必须为 False")
    if (
        permission_decision["effective_action"]
        != "require_confirmation"
        or permission_decision["decision_reason"]
        != "confirmation_required"
    ):
        raise ValueError("当前权限决定不处于待确认状态")

    tool_name = permission_decision["tool_name"]
    if not isinstance(tool_name, str):
        raise TypeError("tool_name 必须是字符串")
    if not tool_name.strip():
        raise ValueError("tool_name 不能为空")

    identity_fields = (
        "authenticated_user_id",
        "authenticated_tenant_id",
        "authentication_event_id",
    )
    for field_name in identity_fields:
        if permission_decision[field_name] != security_context[field_name]:
            raise ValueError(
                f"permission_decision 的 {field_name} "
                "与当前安全上下文不一致"
            )

    try:
        canonical_arguments = json.dumps(
            tool_arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "tool_arguments 必须能够转换为规范 JSON"
        ) from exc

    arguments_sha256 = hashlib.sha256(
        canonical_arguments.encode("utf-8")
    ).hexdigest()

    return {
        "confirmation_id": confirmation_id,
        "tool_name": tool_name,
        "arguments_sha256": arguments_sha256,
        "user_decision": user_decision,
        "confirmed": user_decision == "confirmed",
        "confirmed_by_user_id": security_context[
            "authenticated_user_id"
        ],
        "authenticated_tenant_id": security_context[
            "authenticated_tenant_id"
        ],
        "authentication_event_id": security_context[
            "authentication_event_id"
        ],
        "record_source": "trusted_confirmation_handler",
    }


def evaluate_confirmed_tool_execution(
    *,
    security_context: dict[str, Any],
    tool_policy: dict[str, Any],
    tool_name: str,
    tool_arguments: dict[str, Any],
    confirmation_record: dict[str, Any],
) -> dict[str, Any]:
    """
    在真实工具执行前重新校验权限与确认绑定关系。

    功能：
        先使用当前安全上下文和当前工具策略重新执行权限门，
        防止确认等待期间工具被停用、租户限制发生变化，
        或当前身份的权限已经被撤销。

        当前策略仍允许后，再重新计算待执行参数的规范 JSON
        指纹，并与确认记录中的工具、参数、用户、租户及
        认证事件逐项比较。只有全部一致且用户明确 confirmed，
        才返回 execute_tool。

        这用于缓解 TOCTOU（Time of Check to Time of Use，
        检查时与使用时不一致）风险。

        本函数只返回执行前决定，不调用真实工具，
        不消费或持久化确认记录，也不处理并发和一次性令牌。

    参数：
        security_context:
            工具执行时的当前可信安全上下文。

        tool_policy:
            执行时从服务端工具注册表重新读取的当前策略。

        tool_name:
            即将实际执行的工具名称。

        tool_arguments:
            即将实际传给工具的完整参数字典。

        confirmation_record:
            build_tool_confirmation_record 返回的确认记录。

    返回值：
        dict[str, Any]:
            工具执行前决定，包含：

            execution_allowed:
                当前是否可以进入真实工具执行器。

            effective_action:
                execute_tool 或 reject。

            decision_reason:
                allowed、current_policy_rejected、
                current_tool_policy_mismatch、
                confirmation_not_confirmed、tool_name_changed、
                tool_arguments_changed、confirmed_user_changed、
                tenant_changed 或 authentication_event_changed。

            tool_name、confirmation_id、arguments_sha256 和
            current_permission_decision：用于内部审计。

    异常：
        TypeError:
            参数或确认字段类型不符合约定时抛出。

        ValueError:
            工具名为空、参数不能规范序列化、确认记录缺字段，
            或确认记录来源不符合约定时抛出。
    """
    if not isinstance(tool_name, str):
        raise TypeError("tool_name 必须是字符串")
    if not tool_name.strip():
        raise ValueError("tool_name 不能为空")
    if not isinstance(tool_arguments, dict):
        raise TypeError("tool_arguments 必须是字典")
    if not isinstance(confirmation_record, dict):
        raise TypeError("confirmation_record 必须是字典")

    current_permission_decision = evaluate_tool_action_permission(
        security_context=security_context,
        tool_policy=tool_policy,
        user_confirmation="confirmed",
    )

    required_confirmation_fields = {
        "confirmation_id",
        "tool_name",
        "arguments_sha256",
        "user_decision",
        "confirmed",
        "confirmed_by_user_id",
        "authenticated_tenant_id",
        "authentication_event_id",
        "record_source",
    }
    missing_confirmation_fields = (
        required_confirmation_fields.difference(confirmation_record)
    )
    if missing_confirmation_fields:
        raise ValueError(
            "confirmation_record 缺少字段："
            f"{', '.join(sorted(missing_confirmation_fields))}"
        )
    if (
        confirmation_record["record_source"]
        != "trusted_confirmation_handler"
    ):
        raise ValueError("确认记录必须来自受信确认处理层")

    confirmation_string_fields = {
        "confirmation_id": confirmation_record["confirmation_id"],
        "tool_name": confirmation_record["tool_name"],
        "arguments_sha256": confirmation_record[
            "arguments_sha256"
        ],
        "user_decision": confirmation_record["user_decision"],
        "confirmed_by_user_id": confirmation_record[
            "confirmed_by_user_id"
        ],
        "authenticated_tenant_id": confirmation_record[
            "authenticated_tenant_id"
        ],
        "authentication_event_id": confirmation_record[
            "authentication_event_id"
        ],
    }
    for field_name, field_value in confirmation_string_fields.items():
        if not isinstance(field_value, str):
            raise TypeError(f"{field_name} 必须是字符串")
        if not field_value.strip():
            raise ValueError(f"{field_name} 不能为空")
    if type(confirmation_record["confirmed"]) is not bool:
        raise TypeError("confirmed 必须是布尔值")

    try:
        canonical_arguments = json.dumps(
            tool_arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "tool_arguments 必须能够转换为规范 JSON"
        ) from exc
    arguments_sha256 = hashlib.sha256(
        canonical_arguments.encode("utf-8")
    ).hexdigest()

    if not current_permission_decision["allowed"]:
        execution_allowed = False
        effective_action = "reject"
        decision_reason = "current_policy_rejected"
    elif current_permission_decision["tool_name"] != tool_name:
        execution_allowed = False
        effective_action = "reject"
        decision_reason = "current_tool_policy_mismatch"
    elif (
        confirmation_record["user_decision"] != "confirmed"
        or confirmation_record["confirmed"] is not True
    ):
        execution_allowed = False
        effective_action = "reject"
        decision_reason = "confirmation_not_confirmed"
    elif confirmation_record["tool_name"] != tool_name:
        execution_allowed = False
        effective_action = "reject"
        decision_reason = "tool_name_changed"
    elif confirmation_record["arguments_sha256"] != arguments_sha256:
        execution_allowed = False
        effective_action = "reject"
        decision_reason = "tool_arguments_changed"
    elif (
        confirmation_record["confirmed_by_user_id"]
        != security_context["authenticated_user_id"]
    ):
        execution_allowed = False
        effective_action = "reject"
        decision_reason = "confirmed_user_changed"
    elif (
        confirmation_record["authenticated_tenant_id"]
        != security_context["authenticated_tenant_id"]
    ):
        execution_allowed = False
        effective_action = "reject"
        decision_reason = "tenant_changed"
    elif (
        confirmation_record["authentication_event_id"]
        != security_context["authentication_event_id"]
    ):
        execution_allowed = False
        effective_action = "reject"
        decision_reason = "authentication_event_changed"
    else:
        execution_allowed = True
        effective_action = "execute_tool"
        decision_reason = "allowed"

    return {
        "execution_allowed": execution_allowed,
        "effective_action": effective_action,
        "decision_reason": decision_reason,
        "tool_name": tool_name,
        "confirmation_id": confirmation_record["confirmation_id"],
        "arguments_sha256": arguments_sha256,
        "current_permission_decision": current_permission_decision,
    }


def evaluate_untrusted_content_use(
    *,
    content_record: dict[str, Any],
    intended_use: str,
) -> dict[str, Any]:
    """
    根据固定策略评估不可信外部内容准备被如何使用。

    功能：
        先确认外部内容仍保持 untrusted、无指令权、
        只按数据处理的安全标签，再根据 intended_use
        返回允许作为数据使用、转交独立授权或直接拒绝。

        extract_facts、summarize、answer_with_citation
        只读取内容中的数据，可以 allow_as_data。

        propose_tool_action 只能形成未经授权的工具提议，
        必须进入独立的工具登记、参数、租户、权限和确认流程，
        不能因为文档中写了命令就直接执行。

        override_system_policy、grant_permission、
        change_tenant_identity 和 reveal_secret 试图让外部内容
        获得控制权或突破安全边界，必须 reject。

        本函数不扫描正文，也不识别自然语言攻击。
        intended_use 应由编排层根据当前准备执行的系统操作提供；
        即使该分类由 LLM 建议，最终决定仍由这里的固定映射产生。

    参数：
        content_record:
            build_untrusted_content_record 创建的外部内容记录。

        intended_use:
            当前系统准备如何使用该内容，只允许：
            extract_facts、summarize、answer_with_citation、
            propose_tool_action、override_system_policy、
            grant_permission、change_tenant_identity、
            reveal_secret。

    返回值：
        dict[str, Any]:
            内容使用决定，包含：

            use_allowed:
                是否允许直接按照 intended_use 使用内容。
                propose_tool_action 为 False，因为它只能转交
                独立授权，不能直接当作已批准动作。

            effective_action:
                allow_as_data、require_independent_authorization
                或 reject。

            decision_reason:
                data_use_allowed、untrusted_action_proposal
                或 untrusted_content_has_no_control_authority。

            content_id、source_type、intended_use、
            source_reference 和 content_sha256：用于内部审计。

    异常：
        TypeError:
            参数或内容记录关键字段类型不正确时抛出。

        ValueError:
            内容记录缺字段、安全标签被破坏、字符串为空，
            或 intended_use 不受支持时抛出。
    """
    if not isinstance(content_record, dict):
        raise TypeError("content_record 必须是字典")
    if not isinstance(intended_use, str):
        raise TypeError("intended_use 必须是字符串")
    if not intended_use.strip():
        raise ValueError("intended_use 不能为空")

    required_content_fields = {
        "content_id",
        "source_type",
        "source_reference",
        "trust_level",
        "instruction_authority",
        "handling_action",
        "content_sha256",
    }
    missing_content_fields = required_content_fields.difference(
        content_record
    )
    if missing_content_fields:
        raise ValueError(
            "content_record 缺少字段："
            f"{', '.join(sorted(missing_content_fields))}"
        )

    content_string_fields = {
        "content_id": content_record["content_id"],
        "source_type": content_record["source_type"],
        "source_reference": content_record["source_reference"],
        "content_sha256": content_record["content_sha256"],
    }
    for field_name, field_value in content_string_fields.items():
        if not isinstance(field_value, str):
            raise TypeError(f"{field_name} 必须是字符串")
        if not field_value.strip():
            raise ValueError(f"{field_name} 不能为空")

    if content_record["trust_level"] != "untrusted":
        raise ValueError("外部内容必须保持 untrusted 标签")
    if content_record["instruction_authority"] is not False:
        raise ValueError("外部内容不能拥有指令权")
    if content_record["handling_action"] != "treat_as_data":
        raise ValueError("外部内容必须按数据处理")

    data_use_types = {
        "extract_facts",
        "summarize",
        "answer_with_citation",
    }
    control_attempt_types = {
        "override_system_policy",
        "grant_permission",
        "change_tenant_identity",
        "reveal_secret",
    }
    allowed_use_types = (
        data_use_types
        | control_attempt_types
        | {"propose_tool_action"}
    )
    if intended_use not in allowed_use_types:
        raise ValueError("intended_use 不在允许的使用类型中")

    if intended_use in data_use_types:
        use_allowed = True
        effective_action = "allow_as_data"
        decision_reason = "data_use_allowed"
    elif intended_use == "propose_tool_action":
        use_allowed = False
        effective_action = "require_independent_authorization"
        decision_reason = "untrusted_action_proposal"
    else:
        use_allowed = False
        effective_action = "reject"
        decision_reason = (
            "untrusted_content_has_no_control_authority"
        )

    return {
        "use_allowed": use_allowed,
        "effective_action": effective_action,
        "decision_reason": decision_reason,
        "content_id": content_record["content_id"],
        "source_type": content_record["source_type"],
        "intended_use": intended_use,
        "source_reference": content_record["source_reference"],
        "content_sha256": content_record["content_sha256"],
    }
