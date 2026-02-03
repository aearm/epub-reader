# Complete fixed epub_processor.py

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup, NavigableString
import re
import hashlib
import base64
import os
import spacy
from functools import lru_cache
import threading

# Global SpaCy model cache - loaded once and reused
_nlp_model = None
_nlp_lock = threading.Lock()


def get_spacy_model():
    """Get cached SpaCy model (thread-safe)."""
    global _nlp_model
    if _nlp_model is not None:
        return _nlp_model

    with _nlp_lock:
        # Double-check after acquiring lock
        if _nlp_model is not None:
            return _nlp_model

        # Try models from fastest to slowest
        models = ["en_core_web_sm", "en_core_web_md", "en_core_web_trf"]
        for m in models:
            try:
                print(f"Loading SpaCy model: {m}...")
                nlp = spacy.load(m)
                # Disable components we don't need - but KEEP tok2vec and parser for sentence segmentation!
                disable = [c for c in ["ner", "lemmatizer", "textcat", "attribute_ruler"]
                           if c in nlp.pipe_names]
                if disable:
                    nlp.disable_pipes(disable)
                _nlp_model = nlp
                print(f"SpaCy model {m} loaded successfully (pipes: {nlp.pipe_names})")
                return _nlp_model
            except Exception as e:
                print(f"Could not load {m}: {e}")
                continue

        print("Warning: No SpaCy model available, using regex fallback")
        return None

# Common abbreviations that should NOT be treated as sentence boundaries
ABBREVIATIONS = {
    # Titles
    'Dr', 'Mr', 'Mrs', 'Ms', 'Prof', 'Rev', 'Sr', 'Jr', 'Hon', 'Gov', 'Gen',
    'Col', 'Lt', 'Sgt', 'Capt', 'Adm', 'Rep', 'Sen', 'Pres', 'Supt', 'Fr', 'Br',
    # Common abbreviations
    'vs', 'etc', 'approx', 'dept', 'govt', 'no', 'vol', 'pg', 'ch', 'fig',
    'inc', 'corp', 'ltd', 'co', 'est', 'max', 'min', 'avg', 'qty', 'misc',
    'ref', 'ex', 'assn', 'intl', 'natl', 'tel', 'fax', 'ext',
    # Latin abbreviations
    'e.g', 'i.e', 'cf', 'et al', 'etc', 'viz', 'nb', 'n.b',
    # Locations
    'St', 'Ave', 'Blvd', 'Rd', 'Ln', 'Ct', 'Pl', 'Apt', 'Ste', 'Bldg', 'Mt', 'Ft',
    # Months
    'Jan', 'Feb', 'Mar', 'Apr', 'Jun', 'Jul', 'Aug', 'Sep', 'Sept', 'Oct', 'Nov', 'Dec',
    # Days
    'Mon', 'Tue', 'Tues', 'Wed', 'Thu', 'Thur', 'Thurs', 'Fri', 'Sat', 'Sun',
    # Units
    'ft', 'in', 'mi', 'km', 'kg', 'lb', 'oz', 'hr', 'min', 'sec',
    # Other
    'pp', 'ed', 'eds', 'trans', 'univ', 'acad', 'assoc', 'bros', 'mgr', 'mfg',
}

# Build regex pattern for abbreviations (case-insensitive)
ABBREV_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(a) for a in ABBREVIATIONS) + r')\.\s*$',
    re.IGNORECASE
)


class EPUBProcessor:
    """
    Convert an EPUB into structured chapters with sentences correctly segmented,
    mapped back to HTML, and wrapped safely without breaking inline tags.
    """

    BLOCK_TAGS = ['p', 'li', 'blockquote', 'td', 'th',
                  'h1', 'h2', 'h3', 'h4', 'h5', 'h6']

    def __init__(self, epub_path: str):
        self.epub_path = epub_path
        self.book = epub.read_epub(epub_path)
        self.images = {}
        self._nlp = None

    def process(self):
        book_data = {
            'title': self._get_title(),
            'chapters': [],
            'sentences': []
        }

        self._extract_images()

        for item in self.book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                chapter_data = self._process_chapter(item)
                if chapter_data['sentences']:
                    book_data['chapters'].append(chapter_data)
                    book_data['sentences'].extend(chapter_data['sentences'])

        return book_data

    def _get_title(self) -> str:
        try:
            return self.book.get_metadata('DC', 'title')[0][0]
        except Exception:
            return 'Untitled Book'

    def _extract_images(self):
        for item in self.book.get_items():
            if item.get_type() == ebooklib.ITEM_IMAGE:
                img_name = item.get_name()
                img_content = item.get_content()

                mime_type = 'image/jpeg'
                lower = img_name.lower()
                if lower.endswith('.png'): mime_type = 'image/png'
                elif lower.endswith('.gif'): mime_type = 'image/gif'
                elif lower.endswith('.svg'): mime_type = 'image/svg+xml'

                img_b64 = base64.b64encode(img_content).decode('utf-8')
                self.images[img_name] = f"data:{mime_type};base64,{img_b64}"

    @property
    def nlp(self):
        """Use global cached SpaCy model."""
        if self._nlp is None:
            self._nlp = get_spacy_model()
        return self._nlp

    def _generate_sentence_id(self, text: str, index: int) -> str:
        key = f"{index}:{text.strip()[:100]}"
        return hashlib.md5(key.encode('utf-8')).hexdigest()

    def _is_abbreviation_boundary(self, text: str) -> bool:
        """Check if text ends with a common abbreviation (not a real sentence boundary)."""
        if not text:
            return False
        return bool(ABBREV_PATTERN.search(text))

    def _is_number_context(self, text: str, next_text: str) -> bool:
        """Check if period is part of a number (e.g., '3.14', '$1.50')."""
        if not text or not next_text:
            return False
        # Check if text ends with digit followed by period
        if re.search(r'\d\.$', text):
            # Check if next text starts with a digit
            return bool(re.match(r'^\d', next_text.strip()))
        return False

    def _merge_abbreviation_splits(self, spans: list, text: str) -> list:
        """Merge spans that were incorrectly split on abbreviations."""
        if len(spans) <= 1:
            return spans

        merged = []
        i = 0

        while i < len(spans):
            current = spans[i]

            # Check if this span ends with an abbreviation
            while i < len(spans) - 1:
                current_text = current['text']
                next_span = spans[i + 1]
                next_text = next_span['text']

                should_merge = False

                # Only merge if current sentence is short (likely an abbreviation split)
                # Don't merge if current is already a full sentence (>100 chars)
                if len(current_text) > 100:
                    break

                # Check for abbreviation at end (Dr., Mr., etc.)
                if self._is_abbreviation_boundary(current_text):
                    should_merge = True

                # Check for number context (e.g., "3.14")
                elif self._is_number_context(current_text, next_text):
                    should_merge = True

                # Check for initials (e.g., "J. K." but only if very short)
                elif len(current_text) < 20 and re.search(r'\b[A-Z]\.$', current_text):
                    # Only merge if next starts with capital letter (continuation)
                    if next_text and next_text[0].isupper():
                        should_merge = True

                if should_merge:
                    # Get the text between current end and next start (whitespace)
                    between_start = current['end']
                    between_end = next_span['start']
                    between_text = text[between_start:between_end] if between_start < between_end else ' '

                    # Merge the spans
                    current = {
                        'text': current['text'] + between_text + next_span['text'],
                        'start': current['start'],
                        'end': next_span['end']
                    }
                    i += 1
                else:
                    break

            merged.append(current)
            i += 1

        return merged

    def _split_block_text_into_spans(self, text: str):
        spans = []
        if not text:
            return spans

        # Debug: Show what text we're trying to split
        if "contrarian question" in text.lower():
            print(f"\n{'='*80}")
            print(f"DEBUG: Splitting text block with 'contrarian question'")
            print(f"Text length: {len(text)} chars")
            print(f"First 200 chars: {text[:200]}...")
            print(f"{'='*80}")

        if self.nlp is not None:
            try:
                doc = self.nlp(text)
                raw_spans = []
                sentence_count = 0
                for s in doc.sents:
                    sentence_count += 1
                    start, end = s.start_char, s.end_char
                    while start < end and text[start].isspace(): start += 1
                    while end > start and text[end-1].isspace(): end -= 1

                    seg = text[start:end]
                    if len(seg) < 3 or not any(c.isalpha() for c in seg):
                        continue

                    raw_spans.append({'text': seg, 'start': start, 'end': end})

                # Post-process to merge abbreviation splits
                spans = self._merge_abbreviation_splits(raw_spans, text)

                # Debug: Show each sentence detected
                if "contrarian" in text.lower():
                    print(f"\nTotal sentences detected by SpaCy: {sentence_count}")
                    print(f"After abbreviation merging: {len(spans)}")
                    for idx, span in enumerate(spans):
                        seg = span['text']
                        print(f"\n  Sentence {idx+1}: [{span['start']}:{span['end']}]")
                        print(f"  Text: {seg[:100]}..." if len(seg) > 100 else f"  Text: {seg}")

                return spans
            except Exception as e:
                print(f"SpaCy error: {e}")
                pass

        # Fallback: regex-based splitting with abbreviation awareness
        raw_spans = []
        regex = re.compile(r'.+?(?:[.!?]+["\']?\s+|\Z)', re.S)
        for m in regex.finditer(text):
            start, end = m.start(), m.end()
            while start < end and text[start].isspace(): start += 1
            while end > start and text[end-1].isspace(): end -= 1

            seg = text[start:end]
            if len(seg) < 3 or not any(c.isalpha() for c in seg):
                continue

            raw_spans.append({'text': seg, 'start': start, 'end': end})

        # Post-process to merge abbreviation splits
        spans = self._merge_abbreviation_splits(raw_spans, text)

        return spans

    def _process_chapter(self, item):
        soup = BeautifulSoup(item.get_content(), 'html.parser')

        chapter_title = None
        for tag in ['h1','h2','h3','title']:
            h = soup.find(tag)
            if h:
                chapter_title = h.get_text(strip=True)
                break

        if not chapter_title:
            name = os.path.basename(item.get_name())
            name = os.path.splitext(name)[0].replace('_',' ').replace('-',' ')
            chapter_title = ' '.join(w.capitalize() for w in name.split())

        self._process_chapter_images(soup)

        container = soup.body if soup.body else soup
        chapter_sentences, html = self._annotate_chapter(container)

        cid = hashlib.md5(chapter_title.encode('utf-8')).hexdigest()

        return {'id': cid,'title': chapter_title,'html': str(html),'sentences': chapter_sentences}

    def _annotate_chapter(self, root):
        chapter_sentences = []
        index = 0

        soup = BeautifulSoup(str(root), 'html.parser')

        for block in soup.find_all(self.BLOCK_TAGS):
            text_nodes = []
            for n in block.descendants:
                if isinstance(n, NavigableString) and n.parent.name not in ('script','style'):
                    text_nodes.append(n)

            if not text_nodes:
                continue

            parts, map_ranges = [], []
            cursor = 0
            for n in text_nodes:
                t = str(n)
                parts.append(t)
                start, end = cursor, cursor + len(t)
                map_ranges.append((n,start,end))
                cursor = end

            full_text = ''.join(parts)
            if not full_text.strip(): continue

            spans = self._split_block_text_into_spans(full_text)
            if not spans: continue

            for s in spans:
                sid = self._generate_sentence_id(s['text'], index)
                s['id'] = sid
                s['sentence_index'] = index

                # Debug: Show sentence ID mapping
                if "contrarian" in s['text'].lower():
                    print(f"\n  → Sentence ID: {sid}")
                    print(f"  → Index: {index}")
                    print(f"  → Full text ({len(s['text'])} chars): {s['text'][:150]}...")

                chapter_sentences.append({'id': sid,'text': s['text'],'sentence_index': index})
                index += 1

            for node, ns, ne in map_ranges:
                original = str(node)
                if not original: continue

                overlapping = [s for s in spans if not (s['end'] <= ns or s['start'] >= ne)]
                if not overlapping: continue

                overlapping.sort(key=lambda x: x['start'])

                new_nodes = []
                local_cursor = 0
                nlen = len(original)

                for s in overlapping:
                    ls = max(s['start'], ns) - ns
                    le = min(s['end'], ne) - ns

                    if ls > local_cursor:
                        before = original[local_cursor:ls]
                        if before: new_nodes.append(before)

                    inner = original[ls:le]
                    if inner:
                        tag = soup.new_tag('span')
                        tag['class'] = ['sentence']
                        tag['data-sentence-id'] = s['id']
                        tag.string = inner
                        new_nodes.append(tag)

                    local_cursor = le

                if local_cursor < nlen:
                    after = original[local_cursor:]
                    if after: new_nodes.append(after)

                for new_node in reversed(new_nodes):
                    node.insert_after(new_node)
                node.extract()

        return chapter_sentences, soup

    def _process_chapter_images(self, soup):
        for img in soup.find_all('img'):
            src = img.get('src')
            if not src: continue
            clean = src.replace('../','')
            for path, data in self.images.items():
                base = os.path.basename(path)
                if clean in path or path.endswith(clean) or base == os.path.basename(src):
                    img['src'] = data
                    img['style'] = 'max-width:100%;height:auto;display:block;margin:10px auto;'
                    break
