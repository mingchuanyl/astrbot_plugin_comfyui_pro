import asyncio
import json
import os
import random
import re
import time
from pathlib import Path

import aiohttp
import requests

from astrbot.api import logger

# ✅ 修改后的代码：
DEFAULT_CONNECT_TIMEOUT = 20
DEFAULT_READ_TIMEOUT = 200
DEFAULT_REQUEST_TIMEOUT = (DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT)
DEFAULT_RETRY_TOTAL = 3
DEFAULT_RETRY_BACKOFF = 1.0
LORA_MANIFEST_SUFFIX = ".lora.json"
LORA_CONTEXT_CACHE_SECONDS = 60
# 远端 LoRA 列表缓存：避免每次生成都在协程里同步请求 ComfyUI
REMOTE_LORA_CACHE_SECONDS = 300
REMOTE_LORA_TIMEOUT = (3, 5)
LORA_FILE_EXTENSIONS = {
    ".safetensors",
    ".ckpt",
    ".pt",
    ".pth",
    ".bin",
    ".pt2",
    ".pkl",
    ".sft",
}
# 在 comfyui_api.py 中找到或新增常见的 LoRA 节点 class_type 定义
SUPPORTED_LORA_NODES = {
    "LoraLoader",                  # ComfyUI 原生 LoRA 节点
    "LoraLoaderModelOnly",         # 原生仅模型 LoRA 节点
    "CR Apply LoRA Stack",         # Comfyroll 节点套件
    "CR LoRA Stack",               # Comfyroll 节点套件
    "LoraManager",                 # LoraManager 扩展节点
    "Power Lora Loader (rgthree)", # rgthree 超级 LoRA 节点
    "Power Lora Loader",           # 部分版本的 rgthree 命名
    "Lora Loader Stack (rgthree)", # rgthree LoRA Stack 节点
    "Lora Loader Stack",           # 部分版本的 Stack 命名
    "Easy LoraLoader",             # ComfyUI-Easy-Use 节点
    "Easy LoraStack",              # ComfyUI-Easy-Use 节点
}
LORA_CLEAR_DEFAULTS_TOKEN = "!clear_defaults"
_HTTP_SESSION = None


def _normalize_server_address(server_address):
    server_address = (server_address or "").strip()
    if not server_address:
        return server_address
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", server_address):
        server_address = f"http://{server_address}"
    return server_address.rstrip("/")


def _coerce_timeout(value):
    if value is None or value == "":
        return DEFAULT_REQUEST_TIMEOUT
    if isinstance(value, (int, float)):
        value = max(float(value), 0.1)
        return (min(value, DEFAULT_CONNECT_TIMEOUT), value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        connect_timeout = max(float(value[0]), 0.1)
        read_timeout = max(float(value[1]), 0.1)
        return (connect_timeout, read_timeout)
    return DEFAULT_REQUEST_TIMEOUT


def _build_http_session():
    global _HTTP_SESSION
    if _HTTP_SESSION is not None:
        return _HTTP_SESSION
    session = requests.Session()
    try:
        from requests.adapters import HTTPAdapter
        try:
            from urllib3.util.retry import Retry
        except Exception:
            from requests.packages.urllib3.util.retry import Retry
        retry = Retry(
            total=DEFAULT_RETRY_TOTAL,
            connect=DEFAULT_RETRY_TOTAL,
            read=DEFAULT_RETRY_TOTAL,
            status=DEFAULT_RETRY_TOTAL,
            backoff_factor=DEFAULT_RETRY_BACKOFF,
            status_forcelist=(408, 409, 425, 429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET", "POST"]),
            raise_on_status=False,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
    except Exception:
        pass
    _HTTP_SESSION = session
    return session


def _http_request(method, url, **kwargs):
    session = _build_http_session()
    timeout = _coerce_timeout(kwargs.pop("timeout", None))
    # ✅ 添加 verify=False 以忽略 SSL 证书验证（仅用于开发/内网穿透环境）
    kwargs.setdefault('verify', False) 
    return session.request(method=method, url=url, timeout=timeout, **kwargs)


def _http_get(url, **kwargs):
    return _http_request("GET", url, **kwargs)


def _http_post(url, **kwargs):
    return _http_request("POST", url, **kwargs)


class ComfyUI:
    def __init__(self, config: dict, data_dir: Path = None) -> None:
        """
        Initialize the ComfyUI API client.

        Args:
            config: plugin config dict
            data_dir: persistent plugin data directory
        """
        self.server_address = _normalize_server_address(config.get("server_address", "127.0.0.1:8188"))
        if self.server_address.startswith("http"):
            self.url = self.server_address
        elif "." in self.server_address and ":" not in self.server_address:
            self.url = f"https://{self.server_address}"
        else:
            self.url = f"http://{self.server_address}"

        sub_conf = config.get("sub_config", {})
        self.steps = sub_conf.get("steps", 20)
        self.width = sub_conf.get("width", 768)
        self.height = sub_conf.get("height", 1024)
        self.neg_prompt = sub_conf.get("negative_prompt", "")

        wf_conf = config.get("workflow_settings", {})
        self.wf_filename = wf_conf.get("json_file", "workflow_api.json")
        self.input_id = str(wf_conf.get("input_node_id", "6"))
        self.neg_node_id = str(wf_conf.get("neg_node_id", ""))
        self.output_id = str(wf_conf.get("output_node_id", "9"))
        self.seed_id = None

        llm_conf = config.get("llm_settings", {}) or {}
        lora_conf = llm_conf.get("lora_control", {}) or {}
        self.lora_control_enabled = bool(lora_conf.get("enabled", False))
        self.inject_lora_catalog = bool(lora_conf.get("inject_catalog_into_system_prompt", True))
        self.inject_selected_lora_hints = bool(lora_conf.get("inject_selected_prompt_hints", True))
        self.keep_workflow_defaults_when_selected = bool(
            lora_conf.get("keep_workflow_defaults_when_selected", True)
        )
        try:
            self.max_lora_count = max(1, int(lora_conf.get("max_lora_count", 4) or 4))
        except (TypeError, ValueError):
            self.max_lora_count = 4
        self.lora_scan_roots = self._parse_lora_scan_roots(lora_conf.get("scan_directories", ""))
        try:
            self.max_trigger_options_per_lora = max(
                1, int(lora_conf.get("max_trigger_options_per_lora", 3) or 3)
            )
        except (TypeError, ValueError):
            self.max_trigger_options_per_lora = 3
        try:
            self.trigger_option_preview_length = max(
                24, int(lora_conf.get("trigger_option_preview_length", 80) or 80)
            )
        except (TypeError, ValueError):
            self.trigger_option_preview_length = 80

        # ✅ 新增：读取手动配置的 LoRA 列表 (lora_lines)
        self.manual_lora_lines = config.get("lora_lines", [])
        # ✅ 新增：读取手动配置的 JSON 原始数据 (lora_json_raw)
        self.manual_lora_json_raw = config.get("lora_json_raw", "")

        if data_dir is not None:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = Path(os.path.dirname(os.path.abspath(__file__)))
            logger.warning("[ComfyUI API] data_dir missing, fallback to plugin dir")

        self.workflow_dir = self.data_dir / "workflow"
        self.workflow_path = self.workflow_dir / self.wf_filename
        self._lora_context_cache = None
        self._lora_context_cache_key = None
        self._lora_context_cache_at = 0.0
        self._lora_scan_cache = None
        self._lora_scan_cache_key = None
        self._lora_scan_cache_at = 0.0
        self._remote_lora_cache = None
        self._remote_lora_cache_at = 0.0

        logger.info(
            f"[ComfyUI API] loaded | workflow_dir: {self.workflow_dir} | workflow: {self.wf_filename}"
        )
        if self.lora_scan_roots:
            logger.info(
                "[ComfyUI API] LoRA scan roots: %s",
                ", ".join(str(path) for path in self.lora_scan_roots),
            )
        if self.manual_lora_lines:
            logger.info(f"[ComfyUI API] Manual LoRA lines loaded: {len(self.manual_lora_lines)} entries")

    def reload_config(self, filename: str, input_id: str = None, output_id: str = None, neg_node_id: str = None):
        """Hot-switch workflow without restarting the plugin."""
        self.wf_filename = filename
        self.workflow_path = self.workflow_dir / filename
        self._clear_lora_caches()

        if input_id:
            self.input_id = str(input_id)
        if output_id:
            self.output_id = str(output_id)
        if neg_node_id:
            self.neg_node_id = str(neg_node_id)

        exists = self.workflow_path.exists()
        status = "exists" if exists else "missing"

        logger.info(
            f"[ComfyUI] switch workflow -> {filename} [{status}] | "
            f"Input:{self.input_id} | Neg:{self.neg_node_id} | Output:{self.output_id or 'auto'}"
        )
        return exists, (
            f"已切换至 {filename}，文件状态：{status}\n"
            f"当前节点设置: Positive={self.input_id}, Negative={self.neg_node_id}, Output={self.output_id or '自动'}"
        )

    @staticmethod
    def _parse_lora_scan_roots(value) -> list:
        if isinstance(value, (list, tuple, set)):
            raw_items = list(value)
        else:
            raw_items = re.split(r"[\r\n]+", str(value or ""))

        roots = []
        seen = set()
        for raw in raw_items:
            path_text = str(raw or "").strip().strip('"').strip("'")
            if not path_text:
                continue
            path = Path(path_text).expanduser()
            try:
                path = path.resolve()
            except Exception:
                path = Path(os.path.abspath(str(path)))
            key = os.path.normcase(str(path))
            if key in seen:
                continue
            seen.add(key)
            roots.append(path)
        return roots

    def _clear_lora_caches(self):
        self._lora_context_cache = None
        self._lora_context_cache_key = None
        self._lora_context_cache_at = 0.0
        self._lora_scan_cache = None
        self._lora_scan_cache_key = None
        self._lora_scan_cache_at = 0.0
        self._remote_lora_cache = None
        self._remote_lora_cache_at = 0.0

    @staticmethod
    def _normalize_lora_key(value) -> str:
        if value is None:
            return ""
        return re.sub(r"\s+", "", str(value).replace("\\", "/")).casefold()

    @staticmethod
    def _coerce_lora_strength(value, default: float = 1.0) -> float:
        try:
            if value in ("", None):
                return float(default)
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _format_lora_strength(value: float) -> str:
        number = round(float(value), 4)
        text = f"{number:.4f}".rstrip("0").rstrip(".")
        return text or "0"

    @staticmethod
    def _normalize_prompt_token(value) -> str:
        if value is None:
            return ""
        return re.sub(r"\s+", " ", str(value).strip()).casefold()

    @staticmethod
    def _read_json_file(path: Path):
        for encoding in ("utf-8", "utf-8-sig"):
            try:
                with open(path, "r", encoding=encoding) as f:
                    return json.load(f)
            except UnicodeError:
                continue
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return json.load(f)

    @staticmethod
    def _trim_preview_text(value: str, max_length: int) -> str:
        text = re.sub(r"\s+", " ", str(value or "").strip()).strip(",; ")
        if len(text) <= max_length:
            return text
        return text[: max(0, max_length - 1)].rstrip() + "…"

    @staticmethod
    def _extract_quoted_segments(value: str) -> list:
        text = str(value or "")
        segments = []
        for pattern in (r"“([^”]+)”", r'"([^"]+)"', r"'([^']+)'"):
            segments.extend(match.strip() for match in re.findall(pattern, text) if match.strip())
        return segments

    def _split_trigger_options(self, value) -> list:
        text = re.sub(r"\s+", " ", str(value or "").strip()).strip(",; ")
        if not text:
            return []

        quoted_segments = self._extract_quoted_segments(text)
        if quoted_segments:
            return quoted_segments
        return [text]

    def _expand_prompt_hint_tokens(self, values) -> list:
        expanded = []
        seen = set()
        for raw in values or []:
            pieces = re.split(r"[,;\n，；]+", str(raw or ""))
            if len(pieces) <= 1:
                pieces = [str(raw or "")]

            for piece in pieces:
                text = re.sub(r"\s+", " ", str(piece or "").strip()).strip(",; ")
                normalized = self._normalize_prompt_token(text)
                if not text or not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                expanded.append(text)
        return expanded

    def _iter_lora_lookup_keys(self, entry: dict):
        raw_values = [
            entry.get("display_name"),
            entry.get("name"),
            entry.get("workflow_name"),
            entry.get("loader_name"),
            entry.get("model_name"),
        ] + list(entry.get("aliases", []))

        seen = set()
        for raw in raw_values:
            text = str(raw or "").strip()
            if not text:
                continue

            variants = {text, text.replace("\\", "/")}
            try:
                path_like = Path(text.replace("\\", "/"))
                variants.add(path_like.name)
                variants.add(path_like.stem)
            except Exception:
                pass

            for variant in variants:
                normalized = self._normalize_lora_key(variant)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                yield normalized

    def _build_lora_lookup(self, catalog: dict) -> dict:
        lookup = {}
        for key, entry in catalog.items():
            for normalized in self._iter_lora_lookup_keys(entry):
                lookup.setdefault(normalized, key)
        return lookup

    @staticmethod
    def _build_lora_entry(
        display_name: str,
        *,
        workflow_name: str = "",
        loader_name: str = "",
        default_strength=None,
        default_clip_strength=None,
        default_active: bool = False,
    ) -> dict:
        from pathlib import Path

        # 1. 自动收集各种可能的别名组合 (解决空格/下划线/后缀/大小写不匹配问题)
        raw_names = [display_name, workflow_name, loader_name]
        generated_aliases = []

        for raw_item in raw_names:
            if not raw_item or raw_item.lower() == "none":
                continue
            
            p = Path(raw_item)
            clean_stem = p.stem  # 去掉 .safetensors / .ckpt 后缀的名字
            file_name = p.name   # 纯文件名 (带后缀)
            full_path = str(p)   # 完整路径

            # 衍生多种容错变体
            candidates = [
                full_path,                               # 全路径: po/lora for pony.safetensors
                file_name,                               # 带后缀:  lora.safetensors
                clean_stem,                              # 无后缀: lora
                clean_stem.lower(),                      # 全小写
                clean_stem.replace(" ", "_"),            # 空格转下划线: lora_lora
                clean_stem.replace("_", " "),            # 下划线转空格
                clean_stem.replace(" ", "_").lower(),    # 小写下划线
                clean_stem.replace("_", " ").lower(),    # 小写空格
            ]

            for cand in candidates:
                if cand and cand not in generated_aliases:
                    generated_aliases.append(cand)

        # 2. 返回包含自动别名的字典
        return {
            "name": display_name,
            "display_name": display_name,
            "workflow_name": workflow_name or "",
            "loader_name": loader_name or "",
            "default_strength": default_strength,
            "default_clip_strength": default_clip_strength,
            "default_active": bool(default_active),
            "aliases": generated_aliases,  # 👈 自动构建容错别名池
            "description": "",
            "trigger_words": [],
            "prompt_hints": [],
            "trigger_options": [],
            "source_nodes": [],
            "model_name": "",
            "base_model": "",
            "tags": [],
            "is_style_lora": False,
        }

    def _merge_lora_entry(self, base: dict, incoming: dict):
        if not base.get("display_name") and incoming.get("display_name"):
            base["display_name"] = incoming["display_name"]
            base["name"] = incoming["display_name"]

        if incoming.get("workflow_name") and not base.get("workflow_name"):
            base["workflow_name"] = incoming["workflow_name"]
        if incoming.get("loader_name") and not base.get("loader_name"):
            base["loader_name"] = incoming["loader_name"]

        if incoming.get("default_active"):
            base["default_active"] = True
        if incoming.get("default_strength") not in (None, ""):
            base["default_strength"] = self._coerce_lora_strength(
                incoming.get("default_strength"), base.get("default_strength", 1.0)
            )
        if incoming.get("default_clip_strength") not in (None, ""):
            base["default_clip_strength"] = self._coerce_lora_strength(
                incoming.get("default_clip_strength"), base.get("default_clip_strength", 1.0)
            )

        for field in ("aliases", "trigger_words", "prompt_hints", "trigger_options", "source_nodes", "tags"):
            merged = [
                str(item).strip()
                for item in list(base.get(field, [])) + list(incoming.get(field, []))
                if str(item).strip()
            ]
            base[field] = list(dict.fromkeys(merged))

        if incoming.get("description") and not base.get("description"):
            base["description"] = str(incoming["description"]).strip()
        if incoming.get("model_name") and not base.get("model_name"):
            base["model_name"] = str(incoming["model_name"]).strip()
        if incoming.get("base_model") and not base.get("base_model"):
            base["base_model"] = str(incoming["base_model"]).strip()
        if incoming.get("is_style_lora"):
            base["is_style_lora"] = True

        return base

    def _get_lora_manifest_path(self) -> Path:
        return self.workflow_path.parent / f"{self.workflow_path.stem}{LORA_MANIFEST_SUFFIX}"

    def _find_lora_loader_nodes(self, workflow: dict):
        nodes = []
        for node_id, node_data in workflow.items():
            if not isinstance(node_data, dict):
                continue
            class_type = node_data.get("class_type", "")
            inputs = node_data.get("inputs", {})
            if not isinstance(inputs, dict):
                continue

            # 1. 识别原生 LoraLoader 单节点
            if class_type in ("LoraLoader", "LoraLoaderModelOnly"):
                nodes.append((str(node_id), node_data, "standard"))
            
            # 2. 识别 rgthree 节点 (Power Lora Loader 以及 Lora Loader Stack)
            elif class_type in (
                "Power Lora Loader (rgthree)",
                "Power Lora Loader",
                "Lora Loader Stack (rgthree)",
                "Lora Loader Stack",
            ):
                nodes.append((str(node_id), node_data, "rgthree"))
            
            # 3. 识别其他 Stack 类型 (如 Comfyroll)
            elif "loras" in inputs and isinstance(inputs.get("loras"), dict):
                 nodes.append((str(node_id), node_data, "stack"))

        return nodes
    def _extract_lora_catalog(self, workflow: dict):
        catalog = {}
        loader_node_ids = []

        for item in self._find_lora_loader_nodes(workflow):
            if len(item) == 3:
                node_id, node_data, node_kind = item
            else:
                node_id, node_data, node_kind = item[0], item[1], "stack"

            loader_node_ids.append(node_id)
            inputs = node_data.get("inputs", {})

            if node_kind == "standard":
                raw_name = str(inputs.get("lora_name", "")).strip()
                if not raw_name or raw_name.lower() == "none":
                    name = f"Slot_{node_id}"
                    default_active = False
                else:
                    name = raw_name
                    default_active = True

                entry = catalog.setdefault(
                    name,
                    self._build_lora_entry(
                        name,
                        workflow_name=name,
                        default_strength=self._coerce_lora_strength(inputs.get("strength_model"), 1.0),
                        default_clip_strength=self._coerce_lora_strength(inputs.get("strength_clip"), 1.0),
                        default_active=default_active,
                    ),
                )
                if node_id not in entry["source_nodes"]:
                    entry["source_nodes"].append(node_id)

            elif node_kind == "rgthree":
                # 扫描 rgthree 节点中的 lora_01 / lora_1 等参数
                for key, val in inputs.items():
                    m = re.match(r"^lora_(\d+)$", key, re.IGNORECASE)
                    if not m:
                        continue
                    num_str = m.group(1)
                    raw_name = str(val or "").strip()
                    if not raw_name or raw_name.lower() == "none":
                        continue

                    str_val = inputs.get(f"strength_{num_str}")
                    if str_val is None:
                        str_val = inputs.get(f"lora_{num_str}_strength", 1.0)
                    
                    clip_str_val = inputs.get(f"clip_strength_{num_str}")
                    if clip_str_val is None:
                        clip_str_val = inputs.get(f"lora_{num_str}_clip_strength", str_val)

                    strength = self._coerce_lora_strength(str_val, 1.0)
                    clip_strength = self._coerce_lora_strength(clip_str_val, strength)

                    entry = catalog.setdefault(
                        raw_name,
                        self._build_lora_entry(
                            raw_name,
                            workflow_name=raw_name,
                            default_strength=strength,
                            default_clip_strength=clip_strength,
                            default_active=True,
                        ),
                    )
                    if node_id not in entry["source_nodes"]:
                        entry["source_nodes"].append(node_id)

            else:
                entries = inputs.get("loras", {}).get("__value__", [])
                for entry_item in entries:
                    if not isinstance(entry_item, dict):
                        continue
                    name = str(entry_item.get("name", "")).strip()
                    if not name:
                        continue

                    entry = catalog.setdefault(
                        name,
                        self._build_lora_entry(
                            name,
                            workflow_name=name,
                            default_strength=self._coerce_lora_strength(entry_item.get("strength"), 1.0),
                            default_clip_strength=self._coerce_lora_strength(
                                entry_item.get("clipStrength"), entry_item.get("strength") or 1.0
                            ),
                            default_active=bool(entry_item.get("active", False)),
                        ),
                    )

                    if entry_item.get("active"):
                        entry["default_active"] = True
                    if node_id not in entry["source_nodes"]:
                        entry["source_nodes"].append(node_id)

        return {"catalog": catalog, "loader_node_ids": loader_node_ids}

    def _load_lora_manifest(self) -> dict:
        path = self._get_lora_manifest_path()
        if not path.exists():
            return {}

        try:
            data = self._read_json_file(path)
        except Exception as e:
            logger.warning(f"[ComfyUI] failed to load LoRA sidecar {path.name}: {e}")
            return {}

        if not isinstance(data, dict):
            return {}

        loras = data.get("loras", data)
        return loras if isinstance(loras, dict) else {}

    def _extract_metadata_trigger_options(self, metadata: dict) -> list:
        civitai = metadata.get("civitai", {}) or {}
        raw_values = civitai.get("trainedWords", []) or []

        options = []
        for raw in raw_values:
            options.extend(self._split_trigger_options(raw))

        cleaned = []
        seen = set()
        for item in options:
            text = re.sub(r"\s+", " ", str(item or "").strip()).strip(",; ")
            normalized = self._normalize_prompt_token(text)
            if not text or not normalized or normalized in seen:
                continue
            seen.add(normalized)
            cleaned.append(text)
        return cleaned

    def _scan_lora_catalog(self) -> list:
        cache_key = tuple(str(path) for path in self.lora_scan_roots)
        now = time.monotonic()
        if (
            self._lora_scan_cache_key == cache_key
            and self._lora_scan_cache is not None
            and now - self._lora_scan_cache_at < LORA_CONTEXT_CACHE_SECONDS
        ):
            return self._lora_scan_cache

        entries = []
        seen_files = set()

        for root in self.lora_scan_roots:
            if not root.exists() or not root.is_dir():
                logger.warning(f"[ComfyUI] LoRA scan root does not exist: {root}")
                continue

            for path in root.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in LORA_FILE_EXTENSIONS:
                    continue

                resolved_path = os.path.normcase(str(path.resolve()))
                if resolved_path in seen_files:
                    continue
                seen_files.add(resolved_path)

                try:
                    loader_name = path.relative_to(root).as_posix()
                except ValueError:
                    loader_name = path.name

                metadata_path = path.with_suffix(".metadata.json")
                metadata = {}
                if metadata_path.exists():
                    try:
                        metadata = self._read_json_file(metadata_path)
                    except Exception as e:
                        logger.warning(f"[ComfyUI] failed to read metadata {metadata_path.name}: {e}")
                        metadata = {}

                file_name = str(metadata.get("file_name") or path.stem).strip() or path.stem
                model_name = str(metadata.get("model_name") or "").strip()
                notes = str(metadata.get("notes") or "").strip()
                base_model = str(metadata.get("base_model") or "").strip()
                civitai = metadata.get("civitai", {}) or {}
                civitai_model = civitai.get("model", {}) or {}
                tag_values = []
                for source in (metadata.get("tags", []), civitai_model.get("tags", [])):
                    tag_values.extend(str(item).strip() for item in source if str(item).strip())

                description = notes or model_name or str(civitai_model.get("name") or "").strip()
                trigger_options = self._extract_metadata_trigger_options(metadata)
                aliases = [path.stem]
                if model_name and model_name != file_name:
                    aliases.append(model_name)
                if loader_name and loader_name not in aliases:
                    aliases.append(loader_name)

                tag_text = {self._normalize_prompt_token(tag) for tag in tag_values}
                entry = self._build_lora_entry(file_name, loader_name=loader_name)
                entry["aliases"] = list(dict.fromkeys(alias for alias in aliases if alias))
                entry["description"] = description
                entry["model_name"] = model_name
                entry["base_model"] = base_model
                entry["tags"] = list(dict.fromkeys(tag_values))
                entry["trigger_options"] = trigger_options
                entry["is_style_lora"] = "style" in tag_text or self._normalize_prompt_token(model_name).endswith("/ style")
                entries.append(entry)

        # 如果本地磁盘未扫描到任何 LoRA（例如路径未填/不存在/跨机/内网穿透场景），自动请求远端 ComfyUI API
        if not entries:
            remote_loras = self.fetch_remote_lora_list()
            for lora_name in remote_loras:
                file_name = Path(lora_name).stem
                entry = self._build_lora_entry(
                    file_name,
                    loader_name=lora_name,
                    workflow_name=lora_name
                )
                entry["aliases"] = list(dict.fromkeys([file_name, lora_name]))
                entries.append(entry)

        self._lora_scan_cache = entries
        self._lora_scan_cache_key = cache_key
        self._lora_scan_cache_at = now
        return entries

    def _merge_scanned_lora_catalog(self, catalog: dict):
        if not self.lora_scan_roots:
            return catalog

        lookup = self._build_lora_lookup(catalog)

        for entry in self._scan_lora_catalog():
            matched_key = None
            for normalized in self._iter_lora_lookup_keys(entry):
                matched_key = lookup.get(normalized)
                if matched_key:
                    break

            if matched_key:
                self._merge_lora_entry(catalog[matched_key], entry)
            else:
                unique_key = entry.get("loader_name") or entry.get("display_name")
                suffix = 1
                while unique_key in catalog:
                    suffix += 1
                    unique_key = f"{entry.get('loader_name') or entry.get('display_name')}#{suffix}"
                catalog[unique_key] = entry

            lookup = self._build_lora_lookup(catalog)

        return catalog

    def _apply_manifest_metadata(self, catalog: dict):
        manifest = self._load_lora_manifest()
        if not manifest:
            return catalog

        lookup = self._build_lora_lookup(catalog)
        for raw_name, meta in manifest.items():
            if not isinstance(meta, dict):
                continue

            name = str(raw_name).strip()
            if not name:
                continue

            matched_key = lookup.get(self._normalize_lora_key(name))
            if matched_key:
                entry = catalog[matched_key]
            else:
                entry = self._build_lora_entry(name)
                catalog[name] = entry
                matched_key = name

            aliases = [
                str(alias).strip()
                for alias in meta.get("aliases", [])
                if str(alias).strip()
            ]
            trigger_words = [
                str(word).strip()
                for word in meta.get("trigger_words", [])
                if str(word).strip()
            ]
            prompt_hints = [
                str(word).strip()
                for word in meta.get("prompt_hints", [])
                if str(word).strip()
            ]

            incoming = self._build_lora_entry(
                entry.get("display_name") or name,
                workflow_name=entry.get("workflow_name", ""),
                loader_name=entry.get("loader_name", ""),
                default_strength=self._coerce_lora_strength(
                    meta.get("default_strength"), entry.get("default_strength", 1.0)
                ),
                default_clip_strength=self._coerce_lora_strength(
                    meta.get("default_clip_strength"), entry.get("default_clip_strength", 1.0)
                ),
                default_active=entry.get("default_active", False),
            )
            incoming["aliases"] = aliases
            incoming["description"] = str(meta.get("description") or "").strip()
            incoming["trigger_words"] = trigger_words
            incoming["prompt_hints"] = prompt_hints
            incoming["trigger_options"] = trigger_words
            self._merge_lora_entry(entry, incoming)
            lookup = self._build_lora_lookup(catalog)

        return catalog

        # ✅ 新增：解析手动配置的 lora_lines (支持批量粘贴换行符)
    def _parse_manual_lora_lines(self) -> list:
        entries = []
        if not self.manual_lora_lines:
            return entries
        
        # 兼容处理：如果配置是一个字符串（批量粘贴），按换行符拆分
        lines_to_process = []
        if isinstance(self.manual_lora_lines, str):
            lines_to_process = [line.strip() for line in self.manual_lora_lines.splitlines() if line.strip()]
        elif isinstance(self.manual_lora_lines, list):
            # 如果是列表，遍历每一项，防止某一项内部还包含换行符
            for item in self.manual_lora_lines:
                if isinstance(item, str):
                    lines_to_process.extend([line.strip() for line in item.splitlines() if line.strip()])
                else:
                    lines_to_process.append(str(item))
        
        for line in lines_to_process:
            if not line:
                continue
            
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 2:
                continue
                
            display_name = parts[0]
            file_name = parts[1]
            
            # 提取触发词 (第三部分)
            trigger_words = []
            if len(parts) >= 3 and parts[2]:
                # 支持逗号或中文逗号分隔
                trigger_words = [t.strip() for t in re.split(r"[,，]", parts[2]) if t.strip()]
            
            # 提取备注 (第四部分，可选)
            description = parts[3] if len(parts) >= 4 else ""

            entry = self._build_lora_entry(
                display_name,
                loader_name=file_name,
                workflow_name=file_name
            )
            entry["trigger_words"] = trigger_words
            entry["trigger_options"] = trigger_words # 兼容字段
            entry["description"] = description
            # 手动配置的默认激活
            entry["default_active"] = False 
            
            entries.append(entry)
        
        return entries

    # ✅ 新增：解析手动配置的 lora_json_raw
    def _parse_manual_lora_json(self) -> dict:
        if not self.manual_lora_json_raw or not isinstance(self.manual_lora_json_raw, str):
            return {}
        
        try:
            data = json.loads(self.manual_lora_json_raw)
            if isinstance(data, dict):
                return data
        except Exception as e:
            logger.warning(f"[ComfyUI] Failed to parse manual lora_json_raw: {e}")
        
        return {}

    def get_lora_runtime_context(self) -> dict:
        cache_key = (
            str(self.workflow_path),
            tuple(str(path) for path in self.lora_scan_roots),
            str(self.manual_lora_json_raw)[:50], # 加入手动配置指纹
            str(len(self.manual_lora_lines)),
        )
        now = time.monotonic()
        if (
            self._lora_context_cache_key == cache_key
            and self._lora_context_cache is not None
            and now - self._lora_context_cache_at < LORA_CONTEXT_CACHE_SECONDS
        ):
            return self._lora_context_cache

        try:
            workflow = self._load_workflow()
        except Exception as e:
            logger.warning(f"[ComfyUI] failed to read workflow for LoRA context: {e}")
            return {
                "supported": False,
                "catalog": {},
                "loader_node_ids": [],
                "manifest_path": str(self._get_lora_manifest_path()),
            }

        extracted = self._extract_lora_catalog(workflow)
        catalog = extracted["catalog"]
        self._merge_scanned_lora_catalog(catalog)

        # ==================== ✨ 新增：强行将远端 API 获取到的 LoRA 注入 catalog ====================
        try:
            from pathlib import Path
            remote_loras = self.fetch_remote_lora_list()
            for lora_name in remote_loras:
                # 使用更新后的 _build_lora_entry 构建包含完整别名 (aliases) 的 entry
                entry = self._build_lora_entry(
                    lora_name,
                    loader_name=lora_name,
                    workflow_name=lora_name
                )
                clean_stem = Path(lora_name).stem

                # 检查 catalog 中是否已有对应模型（例如工作流里自带的节点）
                existing_key = None
                for key, cat_entry in catalog.items():
                    if (
                        key == lora_name
                        or key == clean_stem
                        or cat_entry.get("loader_name") == lora_name
                        or lora_name in cat_entry.get("aliases", [])
                        or clean_stem in cat_entry.get("aliases", [])
                    ):
                        existing_key = key
                        break

                if existing_key:
                    # 若已存在，更新真实加载文件名，并合并补充别名
                    catalog[existing_key]["loader_name"] = lora_name
                    merged_aliases = list(dict.fromkeys(
                        catalog[existing_key].get("aliases", []) + entry.get("aliases", [])
                    ))
                    catalog[existing_key]["aliases"] = merged_aliases
                else:
                    # 若不存在，直接作为可用的替换 LoRA 注入 catalog
                    catalog[clean_stem] = entry
        except Exception as e:
            logger.warning(f"[ComfyUI] 注入远端 LoRA 列表至 context 失败: {e}")
        # =========================================================================================

        self._apply_manifest_metadata(catalog)

        # ✅ 新增：合并手动配置的 lora_lines
        manual_entries = self._parse_manual_lora_lines()
        if manual_entries:
            lookup = self._build_lora_lookup(catalog)
            for entry in manual_entries:
                matched_key = None
                for normalized in self._iter_lora_lookup_keys(entry):
                    matched_key = lookup.get(normalized)
                    if matched_key:
                        break
                
                if matched_key:
                    self._merge_lora_entry(catalog[matched_key], entry)
                else:
                    # 如果没找到匹配的，作为新条目添加
                    unique_key = entry.get("loader_name") or entry.get("display_name")
                    catalog[unique_key] = entry
                lookup = self._build_lora_lookup(catalog)

        # ✅ 新增：合并手动配置的 lora_json_raw
        manual_json_data = self._parse_manual_lora_json()
        if manual_json_data:
            lookup = self._build_lora_lookup(catalog)
            for raw_name, meta in manual_json_data.items():
                if not isinstance(meta, dict):
                    continue
                
                name = str(raw_name).strip()
                if not name:
                    continue
                
                matched_key = lookup.get(self._normalize_lora_key(name))
                if matched_key:
                    entry = catalog[matched_key]
                else:
                    entry = self._build_lora_entry(name)
                    catalog[name] = entry
                    matched_key = name
                
                # 合并元数据
                incoming = self._build_lora_entry(
                    entry.get("display_name") or name,
                    workflow_name=entry.get("workflow_name", ""),
                    loader_name=entry.get("loader_name", ""),
                )
                
                # 处理 trigger_words / prompt_hints
                tw = meta.get("trigger_words", [])
                ph = meta.get("prompt_hints", [])
                if tw:
                    incoming["trigger_words"] = [str(t).strip() for t in tw if str(t).strip()]
                    incoming["trigger_options"] = incoming["trigger_words"]
                if ph:
                    incoming["prompt_hints"] = [str(p).strip() for p in ph if str(p).strip()]
                
                self._merge_lora_entry(entry, incoming)
                lookup = self._build_lora_lookup(catalog)


        for entry in catalog.values():
            entry["default_strength"] = self._coerce_lora_strength(entry.get("default_strength"), 1.0)
            entry["default_clip_strength"] = self._coerce_lora_strength(
                entry.get("default_clip_strength"), entry["default_strength"]
            )

        sorted_items = sorted(
            catalog.items(),
            key=lambda item: (
                not bool(item[1].get("default_active")),
                self._normalize_lora_key(item[1].get("display_name") or item[0]),
            ),
        )
        sorted_catalog = {key: value for key, value in sorted_items}
        default_active_loras = [entry for entry in sorted_catalog.values() if entry.get("default_active")]

        context = {
            "supported": bool(extracted["loader_node_ids"]),
            "catalog": sorted_catalog,
            "loader_node_ids": extracted["loader_node_ids"],
            "manifest_path": str(self._get_lora_manifest_path()),
            "scan_roots": [str(path) for path in self.lora_scan_roots],
            "default_active_loras": default_active_loras,
        }
        self._lora_context_cache = context
        self._lora_context_cache_key = cache_key
        self._lora_context_cache_at = now
        return context

    def get_lora_prompt_appendix(self) -> str:
        if not (self.lora_control_enabled and self.inject_lora_catalog):
            return ""

        context = self.get_lora_runtime_context()
        catalog = context.get("catalog", {})
        if not context.get("supported") or not catalog:
            return ""

        default_loras = context.get("default_active_loras", [])
        lines = [
            "--------------------------------------------------",
            "【五、LoRA 使用｜走工具，不写标签】",
            "--------------------------------------------------",
            "",
            "LoRA 现在是一个独立工具：`comfyui_select_lora`。要挂 lora 的话，画图前先调用它，再调用 `comfyui_txt2img`。",
            "",
            "什么时候调用：",
            "- 用户明确要求用某个 LoRA、某个画风 lora，或者点名 lora 名字时",
            "- 用户没提 LoRA，就不要调用，也不要自己挑一个挂上去",
            "",
            "怎么调用：",
            "- `lora_name` 填下面清单里的名字或别名，例如 magakii、chenbin_epoch21",
            "- `strength` 画风类一般 0.6~1.0，默认 1.0；`clip_strength` 不填就跟 strength 一样",
            "- 想取消本轮已选的 LoRA，`lora_name` 传 none",
            "- 名字拿不准就先只调一次、`lora_name` 留空，工具会回一份当前可用清单",
            "",
            f"规则：最多同时挂 {self.max_lora_count} 个。",
            "不要再输出 `<lora picks=\"...\">` 这种标签，插件已经不认了，写了只会把垃圾词塞进提示词。",
            "触发词由工具自动注入，提示词里不用手动复述，也不用照抄清单里的触发词。",
            "",
        ]

        if default_loras:
            lines.append("当前工作流默认启用的 LoRA：")
            for entry in default_loras:
                strength_text = self._format_lora_strength(entry.get("default_strength", 1.0))
                clip_text = self._format_lora_strength(
                    entry.get("default_clip_strength", entry.get("default_strength", 1.0))
                )
                lines.append(f"- {entry['display_name']} | default_on={strength_text}/{clip_text}")
        else:
            lines.append("当前工作流默认启用的 LoRA：无")

        lines.extend(["", "当前可用 LoRA 清单："])
        
        # ✅ 修复：获取所有触发词，不再受 max_trigger_options_per_lora 限制用于展示
        # 注意：为了不让提示词过长，我们依然可以限制预览长度，但必须列出所有索引
        
        for entry in catalog.values():
            strength_text = self._format_lora_strength(entry.get("default_strength", 1.0))
            clip_strength = entry.get("default_clip_strength", entry.get("default_strength", 1.0))
            clip_text = self._format_lora_strength(clip_strength)
            default_flag = "on" if entry.get("default_active") else "off"
            
            # 别名预览保持简短
            alias_values = [
                alias
                for alias in entry.get("aliases", [])
                if self._normalize_lora_key(alias) != self._normalize_lora_key(entry.get("display_name"))
            ]
            alias_preview = ", ".join(alias_values[:2])
            
            # ✅ 核心修改：获取完整的 trigger_options 列表
            all_trigger_options = entry.get("trigger_options", [])
            
            # 为了界面整洁，我们依然可以使用 trim_preview_text，但我们要列出 ALL 选项
            # 如果选项非常多（比如超过10个），可以考虑只显示前10个并标注总数，但通常 LoRA 触发词不会那么多
            display_limit = 15 # 增加展示上限，确保 LLM 能看到更多选项
            
            trigger_preview_parts = []
            for idx, option in enumerate(all_trigger_options, start=1):
                trimmed_opt = self._trim_preview_text(option, self.trigger_option_preview_length)
                trigger_preview_parts.append(f"[{idx}] {trimmed_opt}")
                
            # 如果超过展示上限，截断并提示
            if len(trigger_preview_parts) > display_limit:
                trigger_preview_str = " ; ".join(trigger_preview_parts[:display_limit]) + f" ... (共{len(all_trigger_options)}个)"
            else:
                trigger_preview_str = " ; ".join(trigger_preview_parts)

            details = [f"default={default_flag}:{strength_text}/{clip_text}"]
            if alias_preview:
                details.append(f"aliases={alias_preview}")
            if entry.get("model_name"):
                details.append(f"model={self._trim_preview_text(entry['model_name'], 48)}")
            if entry.get("base_model"):
                details.append(f"base={entry['base_model']}")
                
            if trigger_preview_str:
                details.append(f"triggers={trigger_preview_str}")
            elif entry.get("prompt_hints"):
                details.append(
                    "hints="
                    + " ; ".join(
                        self._trim_preview_text(text, self.trigger_option_preview_length)
                        for text in entry.get("prompt_hints", [])[: self.max_trigger_options_per_lora]
                    )
                )
            elif entry.get("is_style_lora"):
                details.append("triggers=无（更偏全局风格）")
            else:
                details.append("triggers=无")
                
            if entry.get("description"):
                details.append(f"note={self._trim_preview_text(entry['description'], 60)}")
                
            lines.append(f"- {entry['display_name']} | " + " | ".join(details))

        manifest_path = context.get("manifest_path")
        if manifest_path or context.get("scan_roots"):
            lines.extend(
                [
                    "",
                ]
            )
        if context.get("scan_roots"):
            lines.append("LoRA 扫描根目录：")
            for root in context.get("scan_roots", []):
                lines.append(f"- {root}")
        if manifest_path:
            lines.extend(
                [
                    f"工作流 sidecar 元数据文件：{manifest_path}",
                    "如果某个 LoRA 有额外 aliases、trigger_words、prompt_hints、description，请按这些信息理解。",
                ]
            )

        return "\n".join(lines).strip()

    def resolve_lora_selections(self, selections) -> list:
        if not selections:
            return []

        context = self.get_lora_runtime_context()
        catalog = context.get("catalog", {})
        if not context.get("supported") or not catalog:
            return []

        lookup = self._build_lora_lookup(catalog)

        resolved = []
        seen = set()
        clear_defaults = False

        for item in selections:
            if not isinstance(item, dict):
                continue

            raw_name = str(item.get("name", "")).strip()
            if not raw_name:
                continue

            normalized_name = self._normalize_lora_key(raw_name)
            if item.get("control") == "clear_defaults" or normalized_name == self._normalize_lora_key(LORA_CLEAR_DEFAULTS_TOKEN):
                clear_defaults = True
                continue

            matched_key = lookup.get(normalized_name)
            if not matched_key:
                logger.info(f"[ComfyUI] LoRA not found in current workflow: {raw_name}")
                continue
            if matched_key in seen:
                continue

            entry = catalog[matched_key]
            strength = self._coerce_lora_strength(item.get("strength"), entry.get("default_strength", 1.0))
            clip_strength = self._coerce_lora_strength(
                item.get("clip_strength"), entry.get("default_clip_strength", strength)
            )

            # 防御 None 字段，确保拿到 list
            trigger_options = entry.get("trigger_options") or []
            valid_trigger_indexes = []
            selected_prompt_hints = []

            # ✅ 修复核心：正确解析 trigger_indexes
            # 上游 _parse_lora_pick_string 返回的 item['trigger_indexes'] 应该是一个 list [1, 2, 3]
            # 但如果由于某些原因传进来的是字符串 "1,2,3"，我们需要兼容处理
            raw_indexes = item.get("trigger_indexes")
            
            if raw_indexes is not None:
                # 情况 A: 已经是列表 (理想情况)
                if isinstance(raw_indexes, list):
                    for idx in raw_indexes:
                        try:
                            idx_int = int(idx)
                            if 1 <= idx_int <= len(trigger_options):
                                if idx_int not in valid_trigger_indexes:
                                    valid_trigger_indexes.append(idx_int)
                        except (TypeError, ValueError):
                            continue
                
                # 情况 B: 是字符串 (兼容旧逻辑或异常输入)
                elif isinstance(raw_indexes, str):
                    index_str = raw_indexes.strip()
                    if index_str:
                        # 支持逗号、加号、空格分隔
                        parts = re.split(r"[,+\s]+", index_str)
                        for part in parts:
                            part = part.strip()
                            if not part:
                                continue
                            # 处理范围 1-5
                            if "-" in part and part.count("-") == 1:
                                try:
                                    start_str, end_str = part.split("-", 1)
                                    start_idx = int(start_str.strip())
                                    end_idx = int(end_str.strip())
                                    step = 1 if start_idx <= end_idx else -1
                                    for idx in range(start_idx, end_idx + step, step):
                                        if 1 <= idx <= len(trigger_options):
                                            if idx not in valid_trigger_indexes:
                                                valid_trigger_indexes.append(idx)
                                except ValueError:
                                    continue
                            else:
                                # 单个数字
                                try:
                                    trigger_index = int(part)
                                    if 1 <= trigger_index <= len(trigger_options):
                                        if trigger_index not in valid_trigger_indexes:
                                            valid_trigger_indexes.append(trigger_index)
                                except (TypeError, ValueError):
                                    continue

            # ✅ 修复核心：根据解析出的 ALL 有效索引获取提示词，不再限制数量
            # 之前可能只取了前几个，现在全部取出
            for idx in sorted(valid_trigger_indexes):
                # 索引是从 1 开始的，列表是从 0 开始的
                hint = trigger_options[idx - 1]
                if hint and hint not in selected_prompt_hints:
                    selected_prompt_hints.append(hint)

            # 如果没有指定索引，但有 prompt_hints，则使用默认的 hints
            if not selected_prompt_hints:
                prompt_hints = entry.get("prompt_hints") or []
                if prompt_hints:
                    selected_prompt_hints = list(prompt_hints)
                elif len(trigger_options) == 1:
                    selected_prompt_hints = [trigger_options[0]]

            resolved.append(
                {
                    "name": entry.get("workflow_name") or entry.get("loader_name") or entry.get("display_name"),
                    "display_name": entry.get("display_name") or raw_name,
                    "workflow_name": entry.get("workflow_name") or "",
                    "loader_name": entry.get("loader_name") or "",
                    "requested_name": raw_name,
                    "strength": strength,
                    "clip_strength": clip_strength,
                    "aliases": entry.get("aliases") or [],
                    "trigger_words": entry.get("trigger_words") or [],
                    "prompt_hints": entry.get("prompt_hints") or [],
                    "trigger_options": trigger_options,
                    "trigger_indexes": valid_trigger_indexes,
                    "selected_prompt_hints": selected_prompt_hints,
                    "description": entry.get("description", ""),
                }
            )
            seen.add(matched_key)

            max_loras = getattr(self, "max_lora_count", 5)
            if len(resolved) >= max_loras:
                break

        if clear_defaults:
            resolved.insert(0, {"name": LORA_CLEAR_DEFAULTS_TOKEN, "control": "clear_defaults"})

        return resolved

    def _inject_selected_lora_prompt_hints(self, prompt: str, selections: list):
        if not (self.inject_selected_lora_hints and selections):
            return prompt, []

        existing_tokens = {
            self._normalize_prompt_token(token)
            for token in str(prompt or "").split(",")
            if str(token).strip()
        }

        injected = []
        for item in selections:
            if item.get("control"):
                continue
            selected_tokens = self._expand_prompt_hint_tokens(
                item.get("selected_prompt_hints", []) or item.get("prompt_hints", [])
            )
            for token in selected_tokens:
                cleaned = str(token).strip()
                normalized = self._normalize_prompt_token(cleaned)
                if not cleaned or not normalized or normalized in existing_tokens:
                    continue
                injected.append(cleaned)
                existing_tokens.add(normalized)

        if not injected:
            return prompt, []

        prefix = ", ".join(injected)
        final_prompt = f"{prefix}, {str(prompt or '').strip()}".strip(", ").strip()
        return final_prompt, injected

    def _rebuild_lora_text(self, items: list, active_only: bool = True) -> str:
        tokens = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if active_only and not item.get("active", False):
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                continue

            strength = self._format_lora_strength(self._coerce_lora_strength(item.get("strength"), 1.0))
            clip_strength = self._format_lora_strength(
                self._coerce_lora_strength(item.get("clipStrength"), item.get("strength") or 1.0)
            )
            if strength == clip_strength:
                tokens.append(f"<lora:{name}:{strength}>")
            else:
                tokens.append(f"<lora:{name}:{strength}:{clip_strength}>")

        return ", ".join(tokens)

    def _apply_lora_selections(self, workflow: dict, selections: list):
            if not selections:
                return []

            loader_nodes = self._find_lora_loader_nodes(workflow)
            if not loader_nodes:
                logger.warning("[ComfyUI] No compatible LoRA loader nodes found.")
                return []

            clear_defaults = any(item.get("control") == "clear_defaults" for item in selections if isinstance(item, dict))
            effective_selections = [item for item in selections if isinstance(item, dict) and not item.get("control")]

            applied = []
            
            std_nodes = [n for n in loader_nodes if n[2] == "standard"]
            rgthree_nodes = [n for n in loader_nodes if n[2] == "rgthree"]
            stack_nodes = [n for n in loader_nodes if n[2] == "stack"]

            # --- 处理原生 LoraLoader / LoraLoaderModelOnly ---
            if std_nodes:
                for idx, (node_id, node_data, _) in enumerate(std_nodes):
                    inputs = node_data.get("inputs", {})
                    if idx < len(effective_selections) and not clear_defaults:
                        sel = effective_selections[idx]
                        runtime_name = sel.get("loader_name") or sel.get("workflow_name") or sel.get("name")
                        if runtime_name:
                            inputs["lora_name"] = str(runtime_name)
                            # 必须确保转换为 float，防止传递非法类型触发 400 错误
                            inputs["strength_model"] = float(sel.get("strength", 1.0))
                            if "strength_clip" in inputs:
                                inputs["strength_clip"] = float(sel.get("clip_strength", sel.get("strength", 1.0)))
                            applied.append(sel.get("display_name") or runtime_name)
                            logger.info(f"[ComfyUI] Applied to Standard Node {node_id}: {runtime_name}")
                    else:
                        if clear_defaults or not self.keep_workflow_defaults_when_selected:
                            inputs["lora_name"] = "None"  # 部分 ComfyUI 版本识别 "None" 或空字符串，安全起见可设为 "None" 或 ""
                            inputs["strength_model"] = 0.0
                            if "strength_clip" in inputs:
                                inputs["strength_clip"] = 0.0

            # --- 处理 rgthree Power Lora Loader / Lora Loader Stack ---
            if rgthree_nodes:
                for _, node_data, _ in rgthree_nodes:
                    inputs = node_data.get("inputs", {})
                    
                    # 自动检测键名风格（是否补零 01、是否为 strength_01 风格）
                    has_padded_keys = any(re.match(r"^lora_\d{2,}$", k) for k in inputs.keys())
                    has_stack_style_strength = any(re.match(r"^strength_\d+$", k) for k in inputs.keys())

                    existing_slots = [k for k in inputs.keys() if re.match(r"^lora_\d+$", k)]
                    max_slots = max(len(existing_slots), self.max_lora_count) if existing_slots else self.max_lora_count

                    for idx_0based in range(max_slots):
                        prefix_idx = idx_0based + 1
                        idx_str = f"{prefix_idx:02d}" if has_padded_keys else str(prefix_idx)
                        lora_key = f"lora_{idx_str}"
                        
                        if not clear_defaults and idx_0based < len(effective_selections):
                            sel = effective_selections[idx_0based]
                            runtime_name = sel.get("loader_name") or sel.get("workflow_name") or sel.get("name")
                            if runtime_name:
                                inputs[lora_key] = str(runtime_name)
                                strength_val = float(sel.get("strength", 1.0))
                                clip_val = float(sel.get("clip_strength", sel.get("strength", 1.0)))

                                if has_stack_style_strength:
                                    inputs[f"strength_{idx_str}"] = strength_val
                                    if f"clip_strength_{idx_str}" in inputs:
                                        inputs[f"clip_strength_{idx_str}"] = clip_val
                                else:
                                    inputs[f"lora_{idx_str}_strength"] = strength_val
                                    inputs[f"lora_{idx_str}_clip_strength"] = clip_val

                                applied.append(sel.get("display_name") or runtime_name)
                        else:
                            # 清空/禁用多余未选中的槽位
                            inputs[lora_key] = "None"
                            if has_stack_style_strength:
                                inputs[f"strength_{idx_str}"] = 0.0
                                if f"clip_strength_{idx_str}" in inputs:
                                    inputs[f"clip_strength_{idx_str}"] = 0.0
                            else:
                                inputs[f"lora_{idx_str}_strength"] = 0.0
                                inputs[f"lora_{idx_str}_clip_strength"] = 0.0

            # --- 处理其他 Stack 类型 (如 Comfyroll) ---
            if stack_nodes:
                selected_lookup = {}
                for selection in effective_selections:
                    for candidate in (selection.get("name"), selection.get("loader_name"), selection.get("display_name")):
                        normalized = self._normalize_lora_key(candidate)
                        if normalized:
                            selected_lookup.setdefault(normalized, selection)

                for _, node_data, _ in stack_nodes:
                    inputs = node_data.get("inputs", {})
                    lora_container = inputs.get("loras", {})
                    items = lora_container.get("__value__", [])
                    if not isinstance(items, list):
                        continue

                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        name = str(item.get("name", "")).strip()
                        if not name:
                            continue
                        
                        selection = selected_lookup.get(self._normalize_lora_key(name))
                        if selection:
                            item["active"] = True
                            item["strength"] = float(selection["strength"])
                            if "clipStrength" in item:
                                item["clipStrength"] = float(selection["clip_strength"])
                            label = selection.get("display_name") or name
                            if label not in applied:
                                applied.append(label)
                        elif clear_defaults or not self.keep_workflow_defaults_when_selected:
                            item["active"] = False
                    
                    inputs["text"] = self._rebuild_lora_text(items, active_only=True)

            return applied


    def _load_workflow(self):
        if not self.workflow_path.exists():
            raise FileNotFoundError(f"工作流文件不存在: {self.workflow_path}")
        with open(self.workflow_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _inject_params(self, workflow, prompt, lora_selections=None):
        """Inject prompt, optional LoRA selections, steps overrides, and seeds into workflow."""
        node = workflow.get(self.input_id)
        if not node:
            logger.error(f"critical: input node id {self.input_id} not found in workflow")
            return

        inputs = node.get("inputs", {})
        target_keys = [
            "text",
            "opt_text",
            "string",
            "text_positive",
            "positive",
            "prompt",
            "wildcard_text",
        ]
        for key in target_keys:
            if key in inputs:
                inputs[key] = prompt
                break

        if self.neg_node_id and self.neg_prompt:
            neg_node = workflow.get(self.neg_node_id)
            if neg_node:
                n_inputs = neg_node.get("inputs", {})
                n_keys = ["text", "string", "negative", "text_negative", "prompt"]
                for n_key in n_keys:
                    if n_key in n_inputs:
                        existing_neg = str(n_inputs.get(n_key, "")).strip()
                        config_neg = self.neg_prompt.strip()

                        if existing_neg and config_neg:
                            n_inputs[n_key] = f"{existing_neg}, {config_neg}"
                        elif config_neg:
                            n_inputs[n_key] = config_neg
                        break

        if self.lora_control_enabled and lora_selections:
            applied = self._apply_lora_selections(workflow, lora_selections)
            if applied:
                logger.info(f"[ComfyUI] applied LoRA selections: {', '.join(applied)}")
            else:
                logger.info("[ComfyUI] LoRA selections detected, but no compatible LoRA stack node was found")

        overrides = self._load_steps_override()
        if overrides:
            count = self._apply_steps_override(workflow, overrides)
            if count > 0:
                override_info = ", ".join([f"{k}:{v}步" for k, v in overrides.items()])
                logger.info(f"[ComfyUI] steps override applied: {override_info} (updated {count} targets)")
            else:
                logger.info("[ComfyUI] steps override configured but no matching reference was found")

        base_seed = random.randint(1, 999999999999999)
        ks_count = 0
        offset = 0

        for _, node_data in workflow.items():
            if not isinstance(node_data, dict):
                continue
            n_inputs = node_data.get("inputs", {})
            if not isinstance(n_inputs, dict):
                continue

            changed = False

            if "seed" in n_inputs:
                n_inputs["seed"] = base_seed + offset
                offset += 1
                changed = True

            if "noise_seed" in n_inputs:
                n_inputs["noise_seed"] = base_seed + offset
                offset += 1
                changed = True

            if changed:
                ks_count += 1

        logger.info(
            f"[ComfyUI] base seed: {base_seed}, updated {ks_count} seed/noise_seed inputs"
        )

    def _load_steps_override(self) -> dict:
        """Load steps override sidecar for the current workflow."""
        try:
            stem = self.workflow_path.stem
            sidecar = self.workflow_path.parent / f"{stem}.steps.json"

            if not sidecar.exists():
                return {}

            with open(sidecar, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                return {}

            result = {}
            for key, value in data.items():
                if isinstance(value, dict) and "steps" in value:
                    steps = value.get("steps")
                    if isinstance(steps, (int, float)) and steps > 0:
                        result[str(key)] = int(steps)
                elif isinstance(value, (int, float)) and value > 0:
                    result[str(key)] = int(value)

            return result

        except Exception as e:
            logger.warning(f"[ComfyUI] failed to read steps override file: {e}")
            return {}

    def _apply_steps_override(self, workflow: dict, overrides: dict):
        """
        Override steps by node id.

        Only targets steps/steps_total fields that reference the requested ParameterBreak nodes.
        """
        if not overrides:
            return 0

        pb_nodes = {}
        for nid, node_data in workflow.items():
            if isinstance(node_data, dict) and node_data.get("class_type") == "ParameterBreak":
                pb_nodes[str(nid)] = node_data

        if not pb_nodes:
            logger.debug("[ComfyUI] no ParameterBreak nodes detected")
            return 0

        valid_overrides = {}
        for pb_id, steps in overrides.items():
            if pb_id in pb_nodes:
                valid_overrides[pb_id] = steps
            else:
                logger.warning(f"[ComfyUI] override target node {pb_id} not found in current workflow")

        if not valid_overrides:
            return 0

        override_count = 0
        steps_keys = ("steps", "steps_total")

        for nid, node_data in workflow.items():
            if not isinstance(node_data, dict):
                continue

            n_inputs = node_data.get("inputs", {})
            if not isinstance(n_inputs, dict):
                continue

            for key in steps_keys:
                if key not in n_inputs:
                    continue

                value = n_inputs[key]
                if isinstance(value, list) and len(value) == 2:
                    ref_node_id = str(value[0])
                    if ref_node_id in valid_overrides:
                        new_steps = valid_overrides[ref_node_id]
                        n_inputs[key] = new_steps
                        override_count += 1
                        logger.debug(f"[ComfyUI] node {nid}.{key}: [{ref_node_id}] -> {new_steps}")

        return override_count

    async def generate(self, prompt, lora_selections=None):
        """Generate an image with the current workflow."""
        client_id = str(random.randint(100000, 999999))
        try:
            workflow = self._load_workflow()
        except Exception as e:
            return None, str(e)

        if self.lora_control_enabled and lora_selections:
            # resolve_lora_selections 内部可能触发同步 HTTP（远端 LoRA 列表），丢线程池避免卡事件循环
            resolved_loras = await asyncio.to_thread(self.resolve_lora_selections, lora_selections)
        else:
            resolved_loras = []
        prompt, injected_hints = self._inject_selected_lora_prompt_hints(str(prompt or ""), resolved_loras)

        if injected_hints:
            logger.info(f"[ComfyUI] injected LoRA prompt hints: {', '.join(injected_hints)}")

        self._inject_params(workflow, prompt, resolved_loras)

        async with aiohttp.ClientSession() as session:
            payload = {"prompt": workflow, "client_id": client_id}
            # ==================== ✨ 新增：打印请求 JSON 方便调试 ====================
            try:
                # 使用 json.dumps 格式化输出，ensure_ascii=False 可以正常显示中文提示词
                debug_json_str = json.dumps(payload, ensure_ascii=False, indent=2)
                logger.info(f"[ComfyUI Debug] 完整请求 JSON 预览:\n{debug_json_str}")
                
                # 如果你想把完整的 JSON 写入本地文件方便用编辑器查看，也可以取消下面这行的注释：
                # Path(self.data_dir / "last_request_debug.json").write_text(debug_json_str, encoding="utf-8")
            except Exception as e:
                logger.warning(f"[ComfyUI Debug] 打印 JSON 失败: {e}")
            # ========================================================================

            try:
                async with session.post(f"{self.url}/prompt", json=payload) as resp:
                    if resp.status != 200:
                        return None, f"连接 ComfyUI 失败: {resp.status}"
                    res_json = await resp.json()
                    prompt_id = res_json.get("prompt_id")
            except Exception as e:
                return None, f"请求报错: {str(e)}"

            for _ in range(300):
                await asyncio.sleep(1)
                try:
                    async with session.get(f"{self.url}/history/{prompt_id}") as h_resp:
                        if h_resp.status != 200:
                            continue
                        history = await h_resp.json()
                except Exception:
                    continue

                if prompt_id in history:
                    outputs = history[prompt_id].get("outputs", {})
                    img_info = None

                    if self.output_id and self.output_id in outputs:
                        node_out = outputs[self.output_id]
                        # 同时提取图片和视频输出
                        media_list = node_out.get("images", []) + node_out.get("gifs", [])
                        if media_list:
                            img_info = media_list[0]

                    if not img_info:
                        for node_out in outputs.values():
                            media_list = node_out.get("images", []) + node_out.get("gifs", [])
                            if media_list:
                                img_info = media_list[0]
                                break

                    if img_info:
                        fname = img_info["filename"]
                        sfolder = img_info["subfolder"]
                        itype = img_info["type"]
                        img_url = f"{self.url}/view?filename={fname}&subfolder={sfolder}&type={itype}"

                        async with session.get(img_url) as img_res:
                            if img_res.status == 200:
                                return await img_res.read(), None
                            return None, "下载图片失败"

                    return None, "工作流执行完成，但未找到输出图片"

            return None, "生成超时"
        
    def fetch_remote_lora_list(self, force_refresh: bool = False) -> list:
        """通过 ComfyUI API 获取远端已安装的所有 LoRA 文件名列表 (适用于内网穿透/远程部署)

        注意：这是同步阻塞调用（requests），不要直接在 async 函数里 await 之前调用，
        在协程里请用 fetch_remote_lora_list_async。
        """
        now = time.time()
        if (
            not force_refresh
            and self._remote_lora_cache is not None
            and now - self._remote_lora_cache_at < REMOTE_LORA_CACHE_SECONDS
        ):
            return list(self._remote_lora_cache)

        url = f"{self.url.rstrip('/')}/object_info/LoraLoader"
        try:
            resp = _http_get(url, timeout=REMOTE_LORA_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                # 从 LoraLoader 节点的输入参数定义中提取远端 ComfyUI 的 lora 文件名列表
                loras = (
                    data.get("LoraLoader", {})
                    .get("input", {})
                    .get("required", {})
                    .get("lora_name", [[]])[0]
                )
                if isinstance(loras, list):
                    result = [str(l) for l in loras if l and str(l).lower() != "none"]
                    self._remote_lora_cache = list(result)
                    self._remote_lora_cache_at = now
                    return result
        except Exception as e:
            logger.warning(f"[ComfyUI] 通过 API 获取远端 LoRA 列表失败: {e}")
        return []

    async def fetch_remote_lora_list_async(self, force_refresh: bool = False) -> list:
        """fetch_remote_lora_list 的异步版本，放到线程池里跑，避免卡住事件循环。"""
        try:
            return await asyncio.to_thread(self.fetch_remote_lora_list, force_refresh)
        except Exception as e:
            logger.warning(f"[ComfyUI] 异步获取远端 LoRA 列表失败: {e}")
            return []
    def _parse_lora_selection_token(self, token: str) -> dict | None:
        """
        解析单个 LoRA 选择令牌，例如 "Name:0.8@1,2" 或 "Name@1-5"
        返回: {"name": str, "strength": float|None, "clip_strength": float|None, "trigger_indexes": list[int]}
        """
        if not token:
            return None

        # 1. 分离触发词索引部分 (@...)
        trigger_indexes = []
        if "@" in token:
            name_part, index_part = token.rsplit("@", 1)
            # 解析索引部分，支持 1,2,3 或 1-5 或 1+2
            for idx_str in re.split(r"[,+\s]+", index_part):
                idx_str = idx_str.strip()
                if not idx_str:
                    continue
                if "-" in idx_str and idx_str.count("-") == 1:
                    try:
                        start, end = idx_str.split("-", 1)
                        s, e = int(start), int(end)
                        step = 1 if s <= e else -1
                        trigger_indexes.extend(range(s, e + step, step))
                    except ValueError:
                        pass
                else:
                    try:
                        trigger_indexes.append(int(idx_str))
                    except ValueError:
                        pass
            token = name_part
        else:
            name_part = token

        # 2. 分离强度部分 (:Strength:ClipStrength)
        strength = None
        clip_strength = None
        # 使用 rsplit 限制分割次数，防止名字里包含冒号
        parts = [p.strip() for p in name_part.rsplit(":", 2)]
        
        name = ""
        if len(parts) == 1:
            name = parts[0]
        elif len(parts) == 2:
            name, s_val = parts
            try: strength = float(s_val) if s_val else None
            except ValueError: pass
        elif len(parts) == 3:
            name, s_val, c_val = parts
            try: strength = float(s_val) if s_val else None
            except ValueError: pass
            try: clip_strength = float(c_val) if c_val else None
            except ValueError: pass
        
        if not name:
            return None
            
        # 过滤无效索引 (<=0)
        trigger_indexes = [i for i in trigger_indexes if i > 0]

        return {
            "name": name,
            "strength": strength,
            "clip_strength": clip_strength,
            "trigger_indexes": sorted(set(trigger_indexes)) # 去重并排序
        }
