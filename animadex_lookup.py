"""Animadex 角色标签检索（为 Anima 出图注入真实角色/变体 tag）。

思路：
1. 先把用户描述拆成「角色本体」和「服装/形态描述」；
2. 用角色本体去 animadex 查，拿到本体条目 + 全部变体；
3. 变体名字里能对上服装描述的，直接给变体触发词；
4. 对不上，就给本体触发词，让 LLM 自己补服装关键词，并且禁止编造变体名。

用法：
    lookup = AnimadexLookup(logger=logger)
    appendix = await lookup.build_appendix(conf, text, provider=provider)
"""

import asyncio
import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path


class AnimadexLookup:
    _DEFAULT_BASE = "https://animadex.net"
    _INTENT_PATTERN = re.compile(
        r"(画|绘|出图|配图|插图|立绘|贴图|来张|生成.{0,4}图|图里|图片|draw|drawing|illustration|picture|image|pic\b|prompt)",
        re.IGNORECASE,
    )
    _LATIN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9'.\- ]{2,40}")
    # 中文服装/形态词 → 英文描述，用来和变体名做匹配，也顺手当补词提示
    _OUTFIT_HINTS = {
        "兔女郎": "bunny girl",
        "兔子": "bunny",
        "兔耳": "bunny ears",
        "泳装": "swimsuit",
        "水着": "swimsuit",
        "死库水": "school swimsuit",
        "女仆": "maid",
        "婚纱": "wedding dress",
        "校服": "school uniform",
        "制服": "uniform",
        "护士": "nurse",
        "和服": "kimono",
        "浴衣": "yukata",
        "猫耳": "cat ears",
        "圣诞": "christmas",
        "新年": "new year",
        "万圣节": "halloween",
        "偶像": "idol",
        "礼服": "dress",
        "晚礼服": "evening dress",
        "旗袍": "china dress",
        "运动服": "sportswear",
        "居家服": "casual",
        "睡衣": "pajama",
        "西装": "suit",
        "军装": "military uniform",
        "魔法少女": "magical girl",
        "花嫁": "wedding dress",
    }
    # 服装词的同义扩展：描述和变体名/tags 用词不一致时也能对上
    _OUTFIT_SYNONYMS = {
        "bunny": ["bunny", "rabbit ears", "leotard", "playboy bunny", "bunny suit"],
        "kimono": ["kimono", "yukata", "japanese clothes", "hakama"],
        "swimsuit": ["swimsuit", "bikini", "school swimsuit", "one-piece swimsuit"],
        "maid": ["maid", "maid headdress", "apron", "frilled apron"],
        "wedding": ["wedding dress", "bridal", "white dress", "veil"],
        "idol": ["idol", "stage costume", "microphone"],
        "nurse": ["nurse", "nurse cap", "syringe"],
        "dress": ["dress", "gown", "evening dress"],
        "uniform": ["uniform", "serafuku", "school uniform", "blazer"],
        "suit": ["suit", "necktie", "formal"],
        "christmas": ["christmas", "santa", "santa costume"],
        "new year": ["new year", "furisode", "hakama"],
        "pajama": ["pajama", "nightgown", "sleepwear"],
        "cat ears": ["cat ears", "nekomimi", "animal ear fluff"],
        "china dress": ["china dress", "cheongsam", "qipao"],
        "sportswear": ["sportswear", "jersey", "track suit"],
        "halloween": ["halloween", "witch hat", "pumpkin"],
        "magical girl": ["magical girl", "frilled dress", "wand"],
    }
    _OUTFIT_STOPWORDS = {
        "girl", "boy", "outfit", "outfits", "costume", "version", "the", "and",
        "with", "her", "his", "for", "from", "wearing", "wear", "style",
    }

    _BANNED_ALIASES = {
        "一个", "一张", "图片", "角色", "这个", "那个", "衣服", "服装", "女生", "女孩",
        "虚拟主播", "帮我", "画图", "插画", "立绘", "主人", "那个角色", "这个角色",
    }

    def __init__(self, logger=None, store_path=None):
        self.logger = logger
        self.store_path = str(store_path) if store_path else ""
        self._learned_aliases = self._load_alias_store()
        self._search_cache = {}
        self._query_cache = {}
        self._appendix_cache = {}
        # 检索失败的查询词: query -> 错误信息。用来区分「服务挂了/维护」和「真的没这个名字」
        self._search_errors = {}

    # ------------------------------------------------------------------
    # 别名自学习：模型抽出来的中文昵称 -> 英文角色名，落盘复用
    # ------------------------------------------------------------------
    def _load_alias_store(self):
        if not self.store_path:
            return {}
        try:
            with open(self.store_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                return {
                    str(key).strip(): str(value).strip()
                    for key, value in data.items()
                    if str(key).strip() and str(value).strip()
                }
        except FileNotFoundError:
            return {}
        except Exception as exc:  # noqa: BLE001
            if self.logger:
                self.logger.warning(f"[ComfyUI] 读取 Animadex 别名缓存失败: {exc}")
        return {}

    def _save_alias_store(self):
        if not self.store_path:
            return
        try:
            target = Path(self.store_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(self._learned_aliases, handle, ensure_ascii=False, indent=2)
            os.replace(tmp, target)
        except Exception as exc:  # noqa: BLE001
            if self.logger:
                self.logger.warning(f"[ComfyUI] 写入 Animadex 别名缓存失败: {exc}")

    def _learn_alias(self, conf, spec):
        if not conf.get("auto_learn_aliases", True):
            return
        alias = str(spec.get("alias") or "").strip()
        query = str(spec.get("query") or "").strip().lower()
        if not self._valid_alias_pair(alias, query):
            return
        if alias in self._learned_aliases:
            return
        if len(self._learned_aliases) >= self._as_int(conf.get("max_learned_aliases"), 300):
            return
        if alias in self._config_alias_keys(conf):
            return
        self._learned_aliases[alias] = query
        self._save_alias_store()
        if self.logger:
            self.logger.info(f"[ComfyUI] 📝 已记住角色别名: {alias} -> {query}")

    @classmethod
    def _valid_alias_pair(cls, alias, query):
        if not alias or not query:
            return False
        if len(alias) < 2 or len(alias) > 12:
            return False
        if any(ch in alias for ch in "=|\n\r,，。:："):
            return False
        if alias.isascii() or alias in cls._BANNED_ALIASES:
            return False
        if not re.fullmatch(r"[a-z0-9 ()'.,!?&\-]+", query):
            return False
        return True

    @staticmethod
    def _config_alias_keys(conf):
        alias_value = conf.get("aliases", "") or ""
        if isinstance(alias_value, list):
            alias_value = "\n".join(str(item) for item in alias_value)
        keys = set()
        for line in str(alias_value).splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            sep = "=" if "=" in line else ("|" if "|" in line else None)
            if not sep:
                continue
            left = line.split(sep, 1)[0]
            for name in re.split(r"[,，、/]", left):
                name = name.strip()
                if name:
                    keys.add(name)
        return keys

    # ------------------------------------------------------------------
    # 对外入口
    # ------------------------------------------------------------------
    async def build_appendix(self, conf: dict, text: str, provider=None) -> str:
        conf = conf or {}
        if not conf.get("enabled", False):
            return ""
        text = (text or "").strip()
        if len(text) < 2:
            return ""

        cache_ttl = self._as_int(conf.get("cache_ttl"), 900)
        cache_key = (hash(text), len(text))
        cached = self._appendix_cache.get(cache_key)
        if cached and time.time() - cached[0] < cache_ttl:
            return cached[1]

        specs = self._alias_specs(conf, text)
        if not specs:
            latin = self._latin_queries(text)
            specs = [{"query": name, "outfit": self._outfit_terms(text, name)} for name in latin]
            if not specs and conf.get("llm_extract", True) and self._INTENT_PATTERN.search(text):
                specs = await self._llm_specs(conf, text, provider)

        blocks = []
        missed = []
        limit = self._as_int(conf.get("max_queries"), 2)
        for spec in self._dedupe_specs(specs)[:limit]:
            block = await self._query_spec(conf, spec, text)
            if block:
                blocks.append(block)
                self._learn_alias(conf, spec)
            else:
                missed.append(spec["query"])

        if not blocks and conf.get("llm_extract", True):
            alt_specs = await self._llm_specs(conf, text, provider)
            for spec in self._dedupe_specs(alt_specs)[:limit]:
                block = await self._query_spec(conf, spec, text)
                if block:
                    blocks.append(block)
                    self._learn_alias(conf, spec)
                elif spec["query"] not in missed:
                    missed.append(spec["query"])

        if not blocks:
            service_down = any(str(q) in self._search_errors for q in missed)
            appendix = self._render_no_result(missed, service_down=service_down) if missed else ""
            self._appendix_cache[cache_key] = (time.time(), appendix)
            return appendix

        appendix = (
            "【Animadex 角色标签｜本轮已联网检索，以下 tag 真实存在，优先使用】\n"
            + "\n".join(blocks)
            + "\n使用规则：\n"
            "1. 触发词和描述 tags 一律原样照抄：反斜杠和括号别动（例如 suzuran \\(let's carnival!\\) \\(arknights\\)）；描述 tags 有几条就写几条，一条不漏，不删减、不改写、不合并。\n"
            "2. 先认角色本体，再看服装／形态。标了“命中”的变体可以直接用：整条触发词照抄。\n"
            "3. 用了变体触发词，就不要再叠加本体版的服装、发型、配饰 tag；两套混写会融合成四不像。\n"
            "4. 除触发词、描述 tags 和用户明确说过的服装／动作／场景之外，不要自己往 prompt 里加新的角色特征（外形、配饰、兽耳、尾巴、翅膀这类）。变体没覆盖而这些内容是用户明确点名的，才补对应关键词。\n"
            "5. 标了“未命中”时，说明这个角色没有你描述的那套服装：用本体触发词 + 用户说的服装关键词（如 bunny girl, leotard, rabbit ears, playboy bunny），绝对不要编一个不存在的变体名。\n"
            "6. 列表里没出现的角色，直接用外形 tag 描述。\n"
            "7. 全量 tags 之间有明显冲突时（例如 twintails 和 single braid、blue hair 和 white hair、发色或胸围并列冲突、两套互斥的服装／配饰），可以按用户这次要的画面做取舍优化：保留和用户描述一致的那一边，删掉打架的那一项。做了这种取舍必须在回复里顺口告诉用户，例如「这轮把 xx 和 xx 冲突的删了一个，留了 xx」。没冲突就不要动。\n"
        )
        if missed:
            appendix += (
                "\n补充："
                + "、".join(missed[:3])
                + " 这些名字没有在 Animadex 里查到。用户明确指名了它，就直接说没查到，问一句作品名或译名对不对，顺带问要不要改用 Danbooru 风格标签继续画；"
                "只是泛泛描述的话，按外形标签画，别编角色名。\n"
            )
        self._appendix_cache[cache_key] = (time.time(), appendix)
        return appendix

    # ------------------------------------------------------------------
    # 描述拆解：角色本体 + 服装描述
    # ------------------------------------------------------------------
    async def _query_spec(self, conf, spec, text):
        results = await asyncio.to_thread(self._search, conf, spec["query"])
        results = await self._expand_with_siblings(conf, spec["query"], results)
        return self._render_block(conf, spec, results, text)

    def _outfit_terms(self, text, target=""):
        """服装描述：中文映射 + 文本里不属于角色名的英文短语"""
        parts = []
        hint = self._outfit_hint(text)
        if hint:
            parts.append(hint)
        target_words = set(re.findall(r"[a-z']+", (target or "").lower()))
        latin = " ".join(self._latin_queries(text)).lower()
        extra = []
        for word in re.findall(r"[a-z']{3,}", latin):
            if word in target_words or word in self._OUTFIT_STOPWORDS or word in extra:
                continue
            extra.append(word)
        if extra:
            parts.append(" ".join(extra))
        return " ".join(parts).strip()

    def _alias_specs(self, conf, text):
        specs = []
        alias_value = conf.get("aliases", "") or ""
        if isinstance(alias_value, list):
            alias_value = "\n".join(str(item) for item in alias_value)
        for line in str(alias_value).splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            sep = "=" if "=" in line else ("|" if "|" in line else None)
            if not sep:
                continue
            left, right = line.split(sep, 1)
            names = [part.strip() for part in re.split(r"[,，、/]", left) if part.strip()]
            target = right.strip()
            if not target:
                continue
            if any(name and name.lower() in text.lower() for name in names):
                specs.append({"query": target, "outfit": self._outfit_terms(text, target)})
        lowered = text.lower()
        for alias, target in self._learned_aliases.items():
            if alias and alias.lower() in lowered:
                specs.append({"query": target, "outfit": self._outfit_terms(text, target)})
        return specs

    def _outfit_hint(self, text):
        lowered = (text or "").lower()
        words = []
        for cn, en in self._OUTFIT_HINTS.items():
            if cn in text and en not in words:
                words.append(en)
        for token in re.findall(r"[a-z']{3,}", lowered):
            if token in self._OUTFIT_STOPWORDS:
                continue
            if token in ("bunny", "swimsuit", "maid", "kimono", "yukata", "nurse", "suit", "dress", "uniform", "idol", "christmas", "halloween", "pajama"):
                if token not in words:
                    words.append(token)
        return " ".join(words)

    def _latin_queries(self, text):
        found = []
        for match in self._LATIN_PATTERN.finditer(text):
            raw = re.sub(r"\s+", " ", match.group(0)).strip(" .-'")
            tokens = [token for token in raw.split(" ") if len(token) >= 3]
            if not tokens:
                continue
            found.append(" ".join(tokens[:4]))
        return found

    async def _llm_specs(self, conf, text, provider):
        if provider is None:
            return []
        cache_ttl = self._as_int(conf.get("cache_ttl"), 900)
        cache_key = "llm:" + str(hash(text))
        cached = self._query_cache.get(cache_key)
        if cached and time.time() - cached[0] < cache_ttl:
            return cached[1]

        system_prompt = (
            "You extract anime character lookups from a chat message. Split each character into the character "
            "itself and the outfit or form the user asked for. "
            "Output ONLY a JSON array, at most 3 items, each item like "
            '{"character": "laffey (azur lane)", "outfit": "bunny girl", "alias": "拉菲"}. '
            "character must be the common english danbooru name of the base character (with series in parentheses "
            "when needed), never a costume variant. outfit must be lowercase english keywords describing clothes or "
            "form, empty string if none. Translate Chinese or Japanese names and outfit words into english. "
            "alias is the exact original-language word the user used for that character (for example 拉菲 or 雪雪), "
            "empty string if the user already wrote an english name. "
            "If the message mentions no specific anime or vtuber character, output []."
        )
        try:
            response = await provider.text_chat(
                prompt=text[:800],
                contexts=[],
                system_prompt=system_prompt,
            )
            raw = str(getattr(response, "completion_text", "") or "")
        except Exception as exc:  # noqa: BLE001
            if self.logger:
                self.logger.warning(f"[ComfyUI] Animadex 角色名抽取失败: {exc}")
            return []

        specs = []
        match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                data = []
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        character = str(item.get("character") or "").strip().lower()
                        outfit = str(item.get("outfit") or "").strip().lower()
                        alias = str(item.get("alias") or "").strip()
                        if character:
                            specs.append({"query": character, "outfit": outfit, "alias": alias})
                    elif isinstance(item, str) and item.strip():
                        specs.append({"query": item.strip().lower(), "outfit": ""})

        specs = self._dedupe_specs(specs)[:3]
        self._query_cache[cache_key] = (time.time(), specs)
        return specs

    @staticmethod
    def _render_no_result(queries, service_down=False):
        listing = "、".join(str(q) for q in (queries or [])[:3]) or "（无）"
        status = (
            "请求失败，可能正在维护或网络不通"
            if service_down
            else "接口正常，确实没有这个名字"
        )
        return (
            "【Animadex 角色标签｜本轮没查到匹配角色】\n"
            f"检索词: {listing}\n"
            f"服务状态: {status}\n"
            "处理方式：\n"
            "1. 先把结果告诉用户：这个名字在 Animadex 里没查到。服务状态是请求失败时，补一句可能正在维护或网络不通，别把锅甩给用户写错名字。\n"
            "2. 接着问一句要不要改用 Danbooru 风格标签继续画。这一轮先别调用画图工具，等用户回话。\n"
            "3. 用户说可以 → 用角色英文名（作品名放括号里）加外形 tag（发色、瞳色、发型、服装、气质）拼提示词；角色名没把握就只写外形标签，不在正文里编造角色名。\n"
            "4. 用户说不用，或者没回话 → 这一轮就先不画。\n"
            "5. 用户只是在描述画面、没点名是谁 → 不用追问，直接用外形 tag 画。\n"
            "6. 任何情况下都不要编造不存在的角色 tag。\n"
        )

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def _search(self, conf, query):
        cache_ttl = self._as_int(conf.get("cache_ttl"), 900)
        cached = self._search_cache.get(query)
        if cached and time.time() - cached[0] < cache_ttl:
            return cached[1]

        base = (conf.get("api_base") or self._DEFAULT_BASE).rstrip("/")
        url = f"{base}/api/characters/search?" + urllib.parse.urlencode({"q": query})
        results = []
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "Mozilla/5.0 (compatible; AstrBot-ComfyUI-Pro)",
                },
            )
            with urllib.request.urlopen(request, timeout=self._as_int(conf.get("timeout"), 6)) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
            results = payload.get("results") or []
        except Exception as exc:  # noqa: BLE001
            self._search_errors[query] = str(exc) or exc.__class__.__name__
            if self.logger:
                self.logger.warning(f"[ComfyUI] Animadex 检索失败({query}): {exc}")
            return []

        self._search_errors.pop(query, None)
        self._search_cache[query] = (time.time(), results)
        return results

    async def _expand_with_siblings(self, conf, query, results):
        """命中的是变体时，用角色主名再查一次，补齐本体和同角色其余变体"""
        if not results:
            return results
        lowered_query = (query or "").lower()
        keys = []
        copyrights = set()
        for item in results[:3]:
            name = str(item.get("name") or "")
            copyrights.add(str(item.get("copyright") or ""))
            head = re.sub(r"\s*\(.*$", "", name).strip().lower()
            if head and head not in lowered_query and head not in keys:
                keys.append(head)
        merged = list(results)
        seen = {str(item.get("slug") or item.get("trigger") or "") for item in results}
        for key in keys[:1]:
            extra = await asyncio.to_thread(self._search, conf, key)
            for item in extra:
                identity = str(item.get("slug") or item.get("trigger") or "")
                if identity in seen:
                    continue
                name = str(item.get("name") or "").strip().lower()
                if not name.startswith(key):
                    continue
                if str(item.get("copyright") or "") not in copyrights:
                    continue
                seen.add(identity)
                merged.append(item)
        return merged

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------
    @staticmethod
    def _name_core(name):
        return re.sub(r"\s*\(.*$", "", str(name or "")).strip().lower()

    def _ambiguity_note(self, query, results):
        """同名不同作品时给出歧义提示：拿不准就先问用户出处"""
        q_core = self._name_core(query)
        if not q_core:
            return ""
        groups = {}
        for item in results:
            if item.get("is_hidden"):
                continue
            core = self._name_core(item.get("name"))
            if not core:
                continue
            if core != q_core and q_core not in core and core not in q_core:
                continue
            count = self._as_int(item.get("count"), 0)
            if count <= 0:
                continue
            work = str(item.get("copyright_name") or item.get("copyright") or "").strip()
            if not work:
                continue
            groups[work] = max(groups.get(work, 0), count)
        if len(groups) < 2:
            return ""

        total = sum(groups.values()) or 1
        ranked = sorted(groups.items(), key=lambda kv: kv[1], reverse=True)
        top_work, top_count = ranked[0]
        ratio = top_count / total
        listing = "、".join(f"{work}（{count} 图）" for work, count in ranked[:4])
        if ratio >= 0.85:
            return (
                f"同名提示：检索到多个作品的同名角色（{listing}），其中 {top_work} 占绝大多数；"
                f"默认按 {top_work} 处理，出图时顺口确认一句作品名即可。"
            )
        return (
            f"同名歧义：检索到多个作品的同名角色（{listing}），无法判断是哪一个。"
            "这一轮先在回复里问用户“是哪个作品的 XX”，问清楚之前不要输出 <pic>，也不要自己挑一个。"
        )

    def _render_block(self, conf, spec, results, text=""):
        if not results:
            return ""
        query = spec.get("query", "")
        outfit = (spec.get("outfit") or "").strip().lower()
        tag_limit = self._as_int(conf.get("max_tags_per_entry"), 0)
        if tag_limit < 0:
            tag_limit = 0
        min_count = self._as_int(conf.get("min_count"), 5)

        candidates = []
        q_core = self._name_core(query)
        for item in results:
            if item.get("is_hidden"):
                continue
            core = self._name_core(item.get("name"))
            if q_core and core and core != q_core and q_core not in core and core not in q_core:
                continue
            count = self._as_int(item.get("count"), 0)
            if count <= 0 or count < min_count:
                continue
            candidates.append(item)
        if not candidates:
            return ""

        strong_words = self._expand_outfit_terms(outfit)

        def score(item):
            """触发词名字 + 条目自带提示词一起算，皮肤名匹配不上时靠提示词兜底"""
            name = str(item.get("name") or "").lower()
            tags = " ".join(str(tag) for tag in (item.get("tags") or [])).lower()
            name_hits = sum(1 for word in strong_words if word in name)
            tag_hits = sum(1 for word in strong_words if word in tags)
            return name_hits, tag_hits

        def hits(item):
            name_hits, tag_hits = score(item)
            return name_hits * 2 + tag_hits

        base_items = [item for item in candidates if self._is_base_entry(item)]
        variant_items = [item for item in candidates if not self._is_base_entry(item)]
        base_items.sort(key=lambda item: self._as_int(item.get("count"), 0), reverse=True)
        variant_items.sort(
            key=lambda item: (hits(item), self._as_int(item.get("count"), 0)),
            reverse=True,
        )

        matched = [item for item in variant_items if strong_words and hits(item) > 0]
        unmatched = [item for item in variant_items if item not in matched]

        variant_limit = self._as_int(conf.get("max_variants"), 16)
        if variant_limit <= 0:
            variant_limit = 16
        chosen = []
        for item in matched[:variant_limit]:
            name_hits, tag_hits = score(item)
            chosen.append((item, "命中" if name_hits else "命中·描述"))
        if base_items:
            chosen.append((base_items[0], "本体"))
        remaining_slots = max(1, variant_limit - len(matched))
        for item in unmatched[:remaining_slots]:
            chosen.append((item, "变体"))

        lines = [f"角色: {query}" + (f" ｜ 描述的服装/形态: {outfit}" if outfit else "")]
        ambiguity = self._ambiguity_note(query, results)
        if ambiguity:
            lines.append(ambiguity)
        if outfit and not matched:
            lines.append(
                f"- 未命中：该角色没有与「{outfit}」对应的现成变体。用下面的本体触发词，再自己补 "
                f"{outfit} 的关键词，不要编造变体名。"
            )
        for item, kind in chosen:
            trigger = str(item.get("trigger") or "").strip()
            escaped = trigger.replace("(", "\\(").replace(")", "\\)")
            tags = []
            for tag in item.get("tags") or []:
                tag = str(tag).strip()
                if tag and tag not in tags:
                    tags.append(tag)
            line = (
                f"- [{kind}] {self._as_int(item.get('count'), 0)} 图"
                f" | 触发词（原样照抄）: {escaped}"
            )
            if tags:
                shown = tags if tag_limit <= 0 else tags[:tag_limit]
                line += " | 描述 tags: " + ", ".join(shown)
            lines.append(line)
        return "\n".join(lines)

    def _expand_outfit_terms(self, outfit):
        terms = []
        for word in re.findall(r"[a-z']+", (outfit or "").lower()):
            if len(word) < 3 or word in self._OUTFIT_STOPWORDS:
                continue
            if word not in terms:
                terms.append(word)
            for key, synonyms in self._OUTFIT_SYNONYMS.items():
                if word == key or word in synonyms:
                    for extra in [key] + synonyms:
                        if extra not in terms:
                            terms.append(extra)
        return terms

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    @staticmethod
    def _is_base_entry(item):
        """本体 = 名字里只有作品名括号，没有任何服装/形态括号"""
        name = str(item.get("name") or "")
        groups = re.findall(r"\(([^()]*)\)", name)
        if not groups:
            return True
        if len(groups) > 1:
            return False
        inner = groups[0].strip().lower()
        for work in (
            str(item.get("copyright") or "").strip().lower().replace("_", " "),
            str(item.get("copyright_name") or "").strip().lower().replace("_", " "),
        ):
            if not work:
                continue
            flat_inner = inner.replace("_", " ")
            if flat_inner == work or flat_inner in work or work in flat_inner:
                return True
        return False

    @staticmethod
    def _dedupe(values):
        seen = set()
        result = []
        for value in values:
            value = str(value or "").strip().lower()
            if not value or value in seen:
                continue
            seen.add(value)
            result.append(value)
        return result

    @classmethod
    def _dedupe_specs(cls, specs):
        seen = set()
        result = []
        for spec in specs or []:
            query = str(spec.get("query") or "").strip().lower()
            outfit = str(spec.get("outfit") or "").strip().lower()
            key = (query, outfit)
            if not query or key in seen:
                continue
            seen.add(key)
            entry = {"query": query, "outfit": outfit}
            alias = str(spec.get("alias") or "").strip()
            if alias:
                entry["alias"] = alias
            result.append(entry)
        return result

    @staticmethod
    def _as_int(value, default):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
