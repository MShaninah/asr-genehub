from __future__ import annotations
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Any, Union
import json

def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


@dataclass
class ASRConfig:
    model_size: str = "large-v3"
    device_preference: str = "auto"
    compute_type_fp16_on_gpu: str = "float16"
    compute_type_int8_on_cpu: str = "int8"
    vad_min_silence_ms: int = 400
    temperature_fallback: tuple = (0.0, 0.2, 0.4)
    beam_size: int = 5
    initial_prompt: str = (
        "بدي، فيني، وين، شلون، صلحلي، سيارتي، تعبانة، شو في هون؟ "
        "البحصة، الصالحية، الحريقة، الشعلان، باب توما، المزة، برزة، باب شرقي، القصاع. "
        "ميكانيكي، كهربجي، كومجي، صاغة، أنتيكا، شاورما، فلافل.")


class SyrianASRKeywordEngine:

    def __init__(self, model_size: str = "large-v3", device_preference: str = "auto"):
        self.cfg = ASRConfig(model_size=model_size, device_preference=device_preference)
        self._whisper_model = None
        self._kw_extractor = None

    def _has_cuda() -> bool:
        try:
            import ctranslate2
            return ctranslate2.get_cuda_device_count() > 0
        except Exception:
            return False

    def _load_whisper(self):
        if self._whisper_model is not None:
            return
        from faster_whisper import WhisperModel

        device = "cuda" if _has_cuda() else "cpu"
        compute_type = (
            self.cfg.compute_type_fp16_on_gpu if _has_cuda() else self.cfg.compute_type_int8_on_cpu
        )
        self._whisper_model = WhisperModel(self.cfg.model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: str, language: str = "ar") -> str:
        self._load_whisper()
        segments, _ = self._whisper_model.transcribe(
            audio_path,
            language=language,
            task="transcribe",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=self.cfg.vad_min_silence_ms),
            beam_size=self.cfg.beam_size,
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=self.cfg.initial_prompt,
        )
        return " ".join(s.text.strip() for s in segments)


    # ==================== TEXT NORMALIZATION ====================
    ARABIC_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
    PHONETIC_CORRECTION = {
    r"\bوريت\b": "اريد",
    r"\bبدي\b": "اريد",
    
    r"\bببجي\b": "بنشرجي",
    r"\bقومجي\b": "كومجي",
    r"\bكهربجي\b": "كهربائي",
    
    r"\bشري\b": "شرقي",
    r"\bباب شري\b": "باب شرقي",
    
    r"\bدوا\b": "دواء",
    r"\bمشفا\b": "مشفى",
    r"\bعندو\b": "عنده",
}
    ALIFS = re.compile(r"[إأٱآ]")
    TEH_MARBUTA = re.compile(r"ة")
    YEH_VARIANTS = re.compile(r"[ىي]")
    MULTI_SPACE = re.compile(r"\s+")
    KEEP_CHARS = re.compile(r"[^0-9A-Za-z\u0600-\u06FF\s]")

    # Enhanced stopwords with Syrian dialect variations
    STOPWORDS = set([
        "و","في","على","من","عن","إلى","الى","هل","أنا","انت","هو","هي","هم","هن",
        "ما","ماذا","كيف","أين","اي","يا","هذا","هذه","ذلك","هناك","مع","ثم","لكن","بل",
        "او","أو","أيضًا","جدا","جدًا","الان","الآن","لو",
        # Syrian dialect stopwords
        "بدك","بدي","فيني","فيك","فيكي","عندي","هون","هنيك","وين","لون","كيفك","شلون","شو","قديش"
    ])

    @classmethod
    def _normalize_ar(cls, text: str) -> str:
        text = text.strip()
        text = re.sub(r"[إأٱآ]", "ا", text)
        text = re.sub(r"[ىي]", "ي", text)
        text = re.sub(r"ة", "ه", text)
        text = re.sub(r"[\u064B-\u0652]", "", text)
        for error, correct in cls.PHONETIC_CORRECTION.items():
            text = re.sub(rf'\b{error}\b', correct, text)
        text = re.sub(r"[^0-9A-Za-z\u0600-\u06FF\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def _light_stem(cls, text: str) -> str:
        s = cls._normalize_ar(text)
        if len(s) <= 3: return s
        prefixes = ["ال", "وال", "بال", "كال", "لل"]
        suffixes = ["ات", "ون", "ين", "ه", "ة"]
        for suf in suffixes:
            if s.endswith(suf) and len(s) > len(suf) + 2: s = s[:-len(suf)]
        for pre in prefixes:
            if s.startswith(pre) and len(s) > len(pre) + 2: s = s[len(pre):]
        return s

    # ==================== SEMANTIC QUERY UNDERSTANDING (NEW) ====================
    
    # Maps user expressions to canonical service categories
    INTENT_PATTERNS = {
        # Car/Auto services
        r"صلح.*سيار[ةت]|سيار[ةت].*صلح|ميكانيك|ورشة|كهرباء.*سيارة|بنشر|تبديل.*زيت": "ورشة تصليح سيارات",
        r"غس[يى]ل.*سيار[ةت]|تلميع.*سيار[ةت]|نظف.*سيار[ةت]": "غسيل وتلميع سيارات",
        r"بنزين|وقود|محطة.*وقود|تعبئة.*بنزين": "محطات وقود",
        r"إطار.*سيار[ةت]|تاير|كاوتش|عجل[ةه].*سيارة": "إطارات وكاوتشوك",
        
        # Medical
        r"دكتور|طبيب|عياد[ةه]|صح[ةه]|علاج|مشفى|مستشفى": "أطباء",
        r"صيدلي[ةه]|دواء|دوا|روشتة|وصفة.*طبية": "صيدليات",
        r"أسنان|سن[ان]|تقويم.*اسنان": "طبيب أسنان",
        r"عيون|نظارة|نظارات|عدسات": "عيون",
        
        # Food
        r"مطعم|أكل|طعام|مطاعم|اكل": "مطاعم",
        r"فلافل|شاورما|سندويش|وجبة.*سريعة|برغر": "وجبات سريعة",
        r"كافيه|قهوة|قهوه|كوفي|كافي": "كافيهات ومقاهي",
        r"حلو[يى]ات|بوظة|آيس.*كريم|كيك|حلو": "حلويات وبوظة",
        r"خبز|مخبز|فرن": "مخابز",
        
        # Education
        r"مدرس|تعليم|درس|خصوصي|تدريس|سكول|مدرسة": "مدرسون خصوصيون",
        r"معهد|لغة|لغات|انجليزي|فرنسي|الماني": "معاهد لغات وكمبيوتر",
        r"تدريب|تطوير|كورس|دورة.*تدريبية": "مراكز تدريب وتطوير",
        
        # Home services
        r"سباكة|سباك|مواسير|\bمي\b|\bماء\b|مياه|صرف.*صحي": "سباكة وصرف صحي",
        r"كهربائي|كهربا|فيش|لمبة|تمديدات.*كهربائية": "كهرباء",
        r"نجار|نجارة|خشب|ابواب|شبابيك": "نجارة وأبواب",
        r"حداد|حدادة|حديد|قفل|كوالين|أبواب.*حديد": "مفاتيح وحدادة",
        r"دهان|صبغ|بويا|جبصين|ديكور.*داخلي": "دهان وجبصين",
        r"تنظيف.*منزل|تنظيف.*بيت|خادمة|شغالة": "تنظيف منازل",
        r"نقل.*أثاث|نقل.*عفش|فك.*تركيب": "خدمات نقل أثاث",
        
        # Shopping
        r"سوبرماركت|بقالة|خضرة|لحمة|مواد.*غذائية": "متاجر غذائية",
        r"ملابس|لابسة|قماش|فستان|بدلة": "ألبسة",
        r"موبايل|تلفون|جوال|هاتف|شاحن": "متاجر إلكترونيات وهواتف",
        r"لابتوب|كمبيوتر|حاسوب|لابتوبات": "متاجر إلكترونيات وحواسيب",
        
        # Professional services
        r"محامي|قانون|استشارة.*قانونية|قضية|محكمة": "محاماة واستشارات قانونية",
        r"محاسب|محاسبة|ضرائب|مالية|تدقيق": "خدمات محاسبة",
        r"تصوير|فوتوغراف|فيديو|كاميرا": "تصوير",
        r"سفر|سياحة|حجز.*طيران|فندق|رحلة": "مكاتب سفر وسياحة",
        
        # Maintenance
        r"تصليح|صيانة|إصلاح|تبديل|تركيب": "صيانة أجهزة عامة",
        r"تكييف|تبريد|تدفئة|مكيف|دفاية": "تكييف وتبريد وتدفئة",
        
        # Beauty
        r"حلاقة|صالون|حلاق|حلاقة.*رجالي": "صالونات حلاقة",
        r"تجميل|عناية.*بشرة|كوافير|صالون.*نسائي": "مراكز تجميل وعناية",
    }

    # Word-level synonyms for normalization before pattern matching
    SYNONYM_MAP = {
        # Verbs → Nouns/Canonical forms
        "صلح": "تصليح",
        "أصلح": "تصليح",
        "يصلح": "تصليح",
        "نصلح": "تصليح",
        "سوي": "تصليح",
        "أسوي": "تصليح",
        "غسل": "غسيل",
        "نظف": "تنظيف",
        "أنظف": "تنظيف",
        "نضف": "تنظيف",
        "درس": "تدريس",
        "علم": "تعليم",
        "أكل": "طعام",
        "اكل": "طعام",
        "شرب": "مشروبات",
        "سوق": "تسوق",
        "شتري": "شراء",
        "أشتري": "شراء",
        
        # Dialect synonyms
        "فيني": "يمكنني",
        "بدي": "أريد",
        "بدك": "تريد",
        "شو": "ما",
        "لون": "أين",
        "وين": "أين",
        "هون": "هنا",
        "هنيك": "هناك",
        "دوا": "دواء",
        "سوا": "عمل",
        "عندي": "لدي",
        
        # Common misspellings/variations
        "ميكانيكي": "ميكانيك",
        "بنشرجي": "بنشر",
        "كهربجي": "كهربائي",
        "صباغ": "دهان",
    }

    @classmethod
    def _expand_synonyms(cls, text: str) -> str:
        """Replace dialect words and verbs with canonical forms"""
        words = text.split()
        expanded = []
        for word in words:
            # Check direct synonym
            if word in cls.SYNONYM_MAP:
                expanded.append(cls.SYNONYM_MAP[word])
            else:
                # Check stemmed version
                stemmed = cls._light_stem(word)
                if stemmed in cls.SYNONYM_MAP:
                    expanded.append(cls.SYNONYM_MAP[stemmed])
                else:
                    expanded.append(word)
        return " ".join(expanded)

    @classmethod
    def _detect_intent(cls, text: str) -> List[str]:
        """Match text against intent patterns to find service categories"""
        normalized = cls._normalize_ar(text)
        expanded = cls._expand_synonyms(normalized)
        
        detected = []
        for pattern, category in cls.INTENT_PATTERNS.items():
            if re.search(pattern, expanded) or re.search(pattern, normalized):
                if category not in detected:
                    detected.append(category)
        return detected

    # ==================== LEXICONS (Enhanced) ====================
    
    AREAS_LEX = {
        "البحصة","الصالحية","الحريقة","الشعلان","باب توما","المزة","المزة فيلات","ركن الدين",
        "برزة","المهاجرين","القصاع","المالكي","الحميدية","سوق مدحت باشا",
        "حلب","حمص","حماة","اللاذقية","طرطوس","السويداء","دير الزور","الرقة","الحسكة",
        "باب شرقي","دمشق", "الشام", "مزة","برزة","الحريقة","البحصة"
    }
    
    PRODUCT_LEX = {
        "لابتوب","لابتوبات","كمبيوتر","موبايل","حاسوب","طابعة","سخان","براد","غسالة","غسالات",
        "شاورما","فلافل","سندويشة","وجبة","منقوشة","بزورية","صاج","زيت","بنزين","وقود",
        "دواء","دوا","أدوية","علاج"
    }
    
    SERVICE_LEX = {
        # Core services
        "دكتور","طبيب","مطعم","تصليح","صيانة","سباكة","نجارة","حدادة","صيدلية","محل",
        "تصليح سيارات","ورشة تصليح سيارات","غسيل وتلميع سيارات","كهربائي سيارات",
        
        # Extended services from your original list...
        "أتمتة المنازل والمتاجر", "أجهزة إنذار وإطفاء", "أحجار وزجاج متخصص", "أحذية وحقائب", "أدوات تصوير", 
        "أدوات منزلية", "أطباء", "أطفال", "أغذية ومستلزمات الحيوانات", "ألبسة", "ألمنيوم", "أنظمة الطاقة الشمسية",
        "أنظمة المراقبة والأمن", "أنف وأذن", "إعدادي", "إيطالي", "ابتدائي", "اختصاصيو التغذية", "استشارات أعمال",
        "الطب البديل والعلاجات التقليدية", "الطباعة", "العلاج الفيزيائي والطبيعي", "العناية القانونية الصحية",
        "المشافي", "بنوك", "بيع سيارات ومركبات", "تركيب ديكورات داخلية", "تصليح وغسيل سجاد", "تصوير",
        "تكييف وتبريد وتدفئة", "تنجيد وستائر", "تنظيف منازل", "ثانوي", "جلدية",
        "حدادة صناعية", "حدادة ونجارة أثاث", "حفر آبار", "حلول ذكية", "حلويات وبوظة", "خدمات الإنترنت",
        "خدمات التسويق والإعلان", "خدمات التصميم", "خدمات التوصيل", "خدمات الجنازة والمقابر", "خدمات برمجية",
        "خدمات تعليمية", "خدمات كهرباء وسباكة", "خدمات محاسبة", "خدمات نقل أثاث", "خياطة وتطريز", "دهان وجبصين",
        "دهان ومواد بناء", "دور رعاية ومراكز إعاقة", "رجالي", "رجالية", "زهور وهدايا", "سباكة وصرف صحي",
        "سوري", "شرقي", "شركات تأمين", "صالات أفراح", "صالات ألعاب وبلياردو", "صالات سينما", "صالونات حلاقة",
        "صيانة أجهزة عامة", "صيانة أجهزة منزلية", "صيانة أجهزة وشبكات", "صيانة دراجات", "صيدليات", "طبيب أسنان",
        "طبيب عام", "عدسات ونظارات", "عطور ومستحضرات تجميل", "عيادات بيطرية", "عيادات تخصصية", "عيون", "غربي",
        "فنادق", "كافيهات ومقاهي", "كهرباء", "متاجر ألعاب وهوايات", "متاجر إلكترونيات وحواسيب",
        "متاجر إلكترونيات وهواتف", "متاجر الأثاث", "متاجر الحيوانات", "متاجر حلويات", "متاجر غذائية", "متاجر قطع غيار",
        "مجوهرات وساعات", "محاماة واستشارات قانونية", "محطات نقل سيارات", "محطات وقود", "محلات الفنون", "مختبرات",
        "مختبرات الأسنان", "مدارس", "مدارس تعليم القيادة", "مدرسون خصوصيون", "مدن ملاهي", "مدني", "مراكز أشعة وتحاليل",
        "مراكز التأهيل والدعم النفسي", "مراكز التدريب والتطوير", "مراكز تجميل وعناية", "مراكز تدريب مهني",
        "مراكز رعاية الأطفال", "مراكز رقص وفنون", "مراكز صيانة سيارات", "مراكز طبية", "مراكز علاج", "مراكز فحص فني",
        "مرشدون سياحيون", "مزارع للإيجار", "مزودو فرص العمل", "مسارح ومراكز ثقافية", "مستلزمات حفلات",
        "مستلزمات رياضية", "مستلزمات زراعية", "مشروبات", "مصورون", "مطابخ", "مطاعم", "معارض ومهرجانات",
        "معاهد لغات وكمبيوتر", "معماري", "مفاتيح وحدادة", "مقاهي إنترنت ودراسة", "مقاولات", "مكاتب الحج والعمرة",
        "مكاتب الشحن الخارجي", "مكاتب الشحن الداخلي", "مكاتب تأشيرات", "مكاتب تاكسي", "مكاتب تخليص معاملات",
        "مكاتب ترجمة وتدقيق", "مكاتب توظيف واستخدام", "مكاتب سفر وسياحة", "مكاتب شحن", "مكاتب صرافة وتحويل أموال",
        "مكاتب عقارات ومقاولات", "مكافحة الحشرات", "مكتبات وقرطاسية", "ملاعب", "منتزهات ومساحات عامة",
        "منسوجات وستائر", "منظمات خيرية ومؤسسات مجتمع مدني", "منظمو حفلات", "مهندسون استشاريون", "مواد بناء وديكور",
        "ميكانيك", "نجارة وأبواب", "نسائي", "نسائية", "نوادي رياضية", "نوادي مكياج", "وجبات سريعة", "ورش حدادة بديلة",
        "وكالات حجز طيران وفنادق"
    }
    
    DEVICE_LEX = {"جهاز","سونار","أشعة","رنين","تخطيط","تخطيط للقلب","ايكو","ماسح","أشعة.*مقطعية"}

    NP_PATTERN = re.compile(r"(?:\b[\u0600-\u06FF]+\b\s*){1,4}")

    @classmethod
    def _fuzzy_match_lexicon(cls, text: str, lexicon: set) -> List[str]:
        norm_text = cls._normalize_ar(text)
        found = []
        
        # Direct regex matching with ه/ة flexibility
        for item in lexicon:
            norm_item = cls._normalize_ar(item)
            # Create pattern that matches ه or ة interchangeably
            pat = re.escape(norm_item).replace("ه", "[هة]").replace("ة", "[هة]")
            pat = re.sub(r"\[هة\]\[هة\]", "[هة]", pat)
            
            if re.search(r'\b' + pat + r'\b', norm_text):
                found.append(item)
        
        # Stem-based matching for words not caught by direct match
        text_tokens = norm_text.split()
        stemmed_tokens = [cls._light_stem(t) for t in text_tokens]
        stemmed_text = " " + " ".join(stemmed_tokens) + " "
        
        for item in lexicon:
            if item in found:
                continue
            
            # Stem the lexicon item
            item_words = cls._normalize_ar(item).split()
            stemmed_item = " ".join([cls._light_stem(w) for w in item_words])
            
            if len(stemmed_item) < 3:
                continue
            
            # Check if stemmed item appears in stemmed text
            if f" {stemmed_item} " in stemmed_text or f" {stemmed_item}ت " in stemmed_text:
                found.append(item)
                
        return list(set(found))

    @classmethod
    def _looks_like_noise(cls, text: str) -> bool:
        t = cls._normalize_ar(text)
        if len(t.split()) <= 1:  # Changed from 2 to 1 to be less aggressive
            return True
        # Expanded noise patterns
        noise_patterns = [
            r"\b(تست|تجربه|تجربة|اختبار|واحد|اثنين|ثلاثه|ثلاثة|مرحبا|السلام|كيف حالك|هلو|هاي|شكرا|تسلم)\b",
            r"^[؟!.,\s]+$"  # Only punctuation
        ]
        for pattern in noise_patterns:
            if re.search(pattern, t):
                return True
        return False

    @classmethod
    def _candidate_phrases(cls, text: str) -> List[str]:
        sents = re.split(r"[\.!\?؟؛\n]+", text)
        cands: List[str] = []
        for s in sents:
            s = s.strip()
            if not s:
                continue
            for m in cls.NP_PATTERN.finditer(s):
                phrase = m.group(0).strip()
                toks = [t for t in phrase.split() if t not in cls.STOPWORDS]
                if 1 <= len(toks) <= 4:
                    cands.append(" ".join(toks))
        return cands

    def _get_yake(self):
        if self._kw_extractor is None:
            import yake
            self._kw_extractor = yake.KeywordExtractor(lan="ar", n=3, top=30)
        return self._kw_extractor

    def _rank_phrases_with_yake(self, doc: str, candidates: List[str], top_k: int = 15) -> List[str]:
        try:
            kw_extractor = self._get_yake()
            scored = kw_extractor.extract_keywords(doc)
            ranked = [p for p, _ in sorted(scored, key=lambda x: x[1])]
            cand_set = set(candidates)
            return [p for p in ranked if p in cand_set][:top_k]
        except Exception:
            # Fallback if yake fails
            return candidates[:top_k]

    @classmethod
    def _tag_phrase(cls, p: str) -> str:
        def is_in_lex(phrase, lex):
            norm_p = cls._normalize_ar(phrase)
            for item in lex:
                norm_item = cls._normalize_ar(item)
                if norm_p == norm_item:
                    return True
                if norm_item in norm_p and len(norm_item) > len(norm_p) * 0.6:
                    return True
            return False

        if is_in_lex(p, cls.AREAS_LEX) or any(tok.startswith(("بال", "في")) for tok in p.split()):
            return "area"
        if is_in_lex(p, cls.DEVICE_LEX):
            return "device"
        if is_in_lex(p, cls.SERVICE_LEX):
            return "service"
        if is_in_lex(p, cls.PRODUCT_LEX):
            return "product"
        return "generic"

    @staticmethod
    def _postprocess_keywords(ranked: List[str]) -> List[str]:
        seen, out = set(), []
        for p in ranked:
            root = p.replace("ال", "")
            if root not in seen and len(p) >= 2:
                seen.add(root)
                out.append(p)
        return out

    # ==================== MAIN EXTRACTION (FIXED) ====================
    
    def extract_keywords(self, text: str, top_k: int = 8) -> Dict[str, List[str]]:
        buckets = {"area": [], "product": [], "service": [], "device": [], "generic": [], "intent": []}
        
        # Step 1: Intent detection (NEW - solves your core problem)
        detected_intents = self._detect_intent(text)
        buckets["intent"] = detected_intents
        
        # Step 2: Direct lexicon matching
        buckets["area"] = self._fuzzy_match_lexicon(text, self.AREAS_LEX)
        buckets["product"] = self._fuzzy_match_lexicon(text, self.PRODUCT_LEX)
        buckets["service"] = self._fuzzy_match_lexicon(text, self.SERVICE_LEX)
        buckets["device"] = self._fuzzy_match_lexicon(text, self.DEVICE_LEX)
        
        # Step 3: YAKE extraction for additional keywords
        clean = self._normalize_ar(text)
        if not self._looks_like_noise(text):
            cands = self._candidate_phrases(clean)
            if cands:
                ranked = self._rank_phrases_with_yake(clean, cands, top_k=top_k)
                yake_items = self._postprocess_keywords(ranked)
                
                known_set = set(buckets["area"] + buckets["product"] + buckets["service"] + buckets["device"])
                known_norm = {self._normalize_ar(k) for k in known_set}
                
                for y in yake_items:
                    y_norm = self._normalize_ar(y)
                    is_redundant = False
                    for k in known_norm:
                        if y_norm in k or k in y_norm:
                            is_redundant = True
                            break
                    
                    if not is_redundant:
                        tag = self._tag_phrase(y)
                        if tag == "generic":
                            buckets["generic"].append(y)
                        else:
                            buckets[tag].append(y)
        
        # Step 4: Deduplication and priority sorting
        # Move intent-detected services to front of service list
        if buckets["intent"]:
            # Add intents to service bucket if they're services
            for intent in buckets["intent"]:
                if intent not in buckets["service"]:
                    buckets["service"].insert(0, intent)
        
        return buckets

    def speech_to_keywords_and_send(self, audio_path: str, url: str) -> Dict[str, object]:
        """Transcribe, extract, and POST to endpoint with retries."""
        import requests
        from requests.adapters import HTTPAdapter
        from requests.packages.urllib3.util.retry import Retry

        text = self.transcribe(audio_path)
        tags = self.extract_keywords(text)
        
        # Build search query: prioritize intent, then specific services
        search_terms = []
        if tags["intent"]:
            search_terms.extend(tags["intent"])
        search_terms.extend(tags["service"])
        search_terms.extend(tags["product"])
        search_terms.extend(tags["area"])
        
        payload = {
            "full_text": text,
            "extracted_data": tags,
            "search_query": " ".join(list(dict.fromkeys(search_terms))),  # Remove duplicates
            "timestamp": "now"
        }
        
        api_status = "pending"
        
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        try:
            resp = session.post(url, json=payload, timeout=(10, 30))
            resp.raise_for_status()
            api_status = "success"
        except requests.exceptions.RequestException as e:
            print(f"API Error: {e}")
            api_status = f"failed: {str(e)}"
            
        return {
            "text": text,
            "tags": tags,
            "search_query": payload["search_query"],
            "api_status": api_status
        }

    def speech_to_keywords(self, audio_path: str) -> Dict[str, object]:
        text = self.transcribe(audio_path)
        tags = self.extract_keywords(text)
        # Priority: intent > service > product > area > device > generic
        flat = (tags["intent"] + tags["service"] + tags["product"] + 
                tags["area"] + tags["device"] + tags["generic"])
        return {
            "text": text, 
            "tags": tags, 
            "flat_keywords": flat[:5],
            "primary_intent": tags["intent"][0] if tags["intent"] else None
        }


# ==================== TRAINING FUNCTIONS (Unchanged) ====================

def convert_hf_whisper_to_ct2(model_name_or_path: str, output_dir: str, quantization: str = "float16"):
    print(f"Converting {model_name_or_path} to CTranslate2 format at {output_dir}...")
    import ctranslate2
    converter = ctranslate2.converters.TransformersConverter(
        model_name_or_path=model_name_or_path,
        load_as_float16=True
    )
    converter.convert(
        output_dir=output_dir,
        quantization=quantization,
        force=True
    )
    print("Conversion complete.")


def train_whisper(
    train_csv: str,
    eval_csv: str,
    output_dir: str,
    model_name: str = "openai/whisper-large-v3",
    epochs: int = 3,
    lr: float = 1e-5,
    per_device_batch_size: int = 4,
    gradient_accumulation_steps: int = 2,
    fp16: bool = True,
    language: str = "Arabic",
) -> None:
    import pandas as pd
    import torch
    from datasets import Dataset, Audio
    from transformers import (
        WhisperFeatureExtractor,
        WhisperTokenizer,
        WhisperProcessor,
        WhisperForConditionalGeneration,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
    )
    from dataclasses import dataclass
    from typing import Any, Dict, List, Union

    print("Loading data...")
    df_train = pd.read_csv(train_csv)
    df_eval = pd.read_csv(eval_csv)

    common_dataset = Dataset.from_dict({"audio": df_train["path"].tolist(), "sentence": df_train["text"].tolist()})
    common_dataset = common_dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    eval_dataset = Dataset.from_dict({"audio": df_eval["path"].tolist(), "sentence": df_eval["text"].tolist()})
    eval_dataset = eval_dataset.cast_column("audio", Audio(sampling_rate=16000))

    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained(model_name, language=language, task="transcribe")
    processor = WhisperProcessor.from_pretrained(model_name, language=language, task="transcribe")

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch

    print("Preprocessing data...")
    common_dataset = common_dataset.map(prepare_dataset, remove_columns=["audio", "sentence"], num_proc=1)
    eval_dataset = eval_dataset.map(prepare_dataset, remove_columns=["audio", "sentence"], num_proc=1)

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any
        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]
                
            batch["labels"] = labels
            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    print("Loading model...")
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        warmup_steps=50,
        max_steps=epochs * (len(common_dataset) // (per_device_batch_size * gradient_accumulation_steps)) + 10,
        gradient_checkpointing=True,
        fp16=fp16,
        evaluation_strategy="steps",
        per_device_eval_batch_size=per_device_batch_size,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=100,
        eval_steps=100,
        logging_steps=25,
        report_to=["none"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
    )

    import evaluate
    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=common_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    print("Starting training...")
    trainer.train()
    
    hf_save_path = os.path.join(output_dir, "hf_best")
    trainer.save_model(hf_save_path)
    processor.save_pretrained(hf_save_path)
    
    ct2_save_path = os.path.join(output_dir, "ct2_fast")
    convert_hf_whisper_to_ct2(hf_save_path, ct2_save_path)
    print(f"\nDONE! Your fast model is ready at: {ct2_save_path}")
    print("To use it:", f'engine = SyrianASRKeywordEngine(model_size="{ct2_save_path}")')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Arabic ASR + Keywords (fast) + optional training")
    sub = parser.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="Transcribe and extract keywords from audio")
    p_run.add_argument("audio", type=str, help="Path to audio file")
    p_run.add_argument("--model_size", type=str, default="large-v3")

    p_train = sub.add_parser("train", help="Fine-tune Whisper and convert to Faster-Whisper")
    p_train.add_argument("--train_csv", required=True)
    p_train.add_argument("--eval_csv", required=True)
    p_train.add_argument("--out", required=True)
    p_train.add_argument("--epochs", type=int, default=3)
    p_train.add_argument("--lr", type=float, default=1e-5)
    p_train.add_argument("--model", type=str, default="openai/whisper-large-v3")

    args = parser.parse_args()

    if args.cmd == "run":
        engine = SyrianASRKeywordEngine(model_size=args.model_size)
        res = engine.speech_to_keywords(args.audio)
        print(json.dumps(res, ensure_ascii=False, indent=2))
    elif args.cmd == "train":
        train_whisper(
            train_csv=args.train_csv,
            eval_csv=args.eval_csv,
            output_dir=args.out,
            model_name=args.model,
            epochs=args.epochs,
            lr=args.lr,
        )
    else:
        parser.print_help()