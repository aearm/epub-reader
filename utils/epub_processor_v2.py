# utils/epub_processor_v2.py
"""
Two-Pass EPUB Processor

Pass 1: Clean HTML - Remove footnotes, references, annotations
Pass 2: Extract & Segment - Use PySBD for accurate sentence boundaries

This approach separates concerns and produces cleaner, more accurate sentences.
"""

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup, NavigableString, Comment
import re
import hashlib
import base64
import os
import pysbd


class EPUBProcessorV2:
    """
    Two-pass EPUB processor for cleaner sentence extraction.

    Pass 1: Clean HTML
        - Remove footnote markers (<sup>, <a class="footnote">, etc.)
        - Remove reference numbers
        - Strip annotations and asides
        - Preserve main content structure

    Pass 2: Extract Sentences
        - Use PySBD for accurate sentence boundary detection
        - PySBD handles abbreviations (Dr., Mr., etc.) correctly
        - Wrap sentences in spans for TTS synchronization
    """

    # Tags that typically contain footnotes/references to remove
    REMOVE_TAGS = ['sup', 'sub']

    # Classes that indicate footnotes/references
    FOOTNOTE_CLASSES = [
        'footnote', 'footnoteref', 'noteref', 'endnote', 'citation',
        'reference', 'ref', 'note', 'annotation', 'aside'
    ]

    # Roles that indicate non-content elements
    FOOTNOTE_ROLES = [
        'doc-noteref', 'doc-footnote', 'doc-endnote', 'note'
    ]

    # Block-level tags to process for sentences
    BLOCK_TAGS = ['p', 'li', 'blockquote', 'td', 'th', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']

    def __init__(self, epub_path: str):
        self.epub_path = epub_path
        self.book = epub.read_epub(epub_path)
        self.images = {}
        self.cover_image = None  # Raw bytes of cover image

        # Initialize PySBD segmenter (handles abbreviations correctly)
        self.segmenter = pysbd.Segmenter(language="en", clean=False)

        # Extract cover image
        self._extract_cover()

    def _extract_cover(self):
        """Extract the cover image from the EPUB."""
        try:
            # Method 1: Look for cover in metadata
            cover_id = None
            for meta in self.book.get_metadata('OPF', 'meta'):
                if meta[1].get('name') == 'cover':
                    cover_id = meta[1].get('content')
                    break

            # Method 2: Look for item with id containing 'cover'
            if not cover_id:
                for item in self.book.get_items():
                    if item.get_type() == ebooklib.ITEM_IMAGE:
                        item_id = item.get_id()
                        item_name = item.get_name().lower()
                        if 'cover' in item_id.lower() or 'cover' in item_name:
                            self.cover_image = item.get_content()
                            return

            # Get cover by ID if found
            if cover_id:
                item = self.book.get_item_with_id(cover_id)
                if item:
                    self.cover_image = item.get_content()
                    return

            # Method 3: Take the first image as cover
            for item in self.book.get_items():
                if item.get_type() == ebooklib.ITEM_IMAGE:
                    self.cover_image = item.get_content()
                    return

        except Exception as e:
            print(f"Error extracting cover: {e}")

    def process(self):
        """Main processing entry point."""
        book_data = {
            'title': self._get_title(),
            'author': self._get_author(),
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
        """Extract book title from metadata."""
        try:
            return self.book.get_metadata('DC', 'title')[0][0]
        except Exception:
            return 'Untitled Book'

    def _get_author(self) -> str:
        """Extract book author from metadata."""
        try:
            return self.book.get_metadata('DC', 'creator')[0][0]
        except Exception:
            return ''

    def _extract_images(self):
        """Extract and encode images as base64 data URIs."""
        for item in self.book.get_items():
            if item.get_type() == ebooklib.ITEM_IMAGE:
                img_name = item.get_name()
                img_content = item.get_content()

                mime_type = 'image/jpeg'
                lower = img_name.lower()
                if lower.endswith('.png'):
                    mime_type = 'image/png'
                elif lower.endswith('.gif'):
                    mime_type = 'image/gif'
                elif lower.endswith('.svg'):
                    mime_type = 'image/svg+xml'

                img_b64 = base64.b64encode(img_content).decode('utf-8')
                self.images[img_name] = f"data:{mime_type};base64,{img_b64}"

    # =========================================================================
    # PASS 1: Clean HTML
    # =========================================================================

    def _clean_html(self, soup: BeautifulSoup) -> BeautifulSoup:
        """
        Pass 1: Remove footnotes, references, and non-content elements.
        """
        # Remove HTML comments
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            comment.extract()

        # Remove script and style tags
        for tag in soup.find_all(['script', 'style', 'noscript']):
            tag.decompose()

        # Remove superscript/subscript tags (common for footnote numbers)
        for tag in soup.find_all(self.REMOVE_TAGS):
            # Check if it contains only a number (likely a reference)
            text = tag.get_text(strip=True)
            if text.isdigit() or re.match(r'^[\d,\s]+$', text):
                tag.decompose()
            # Also remove if it's a link to a footnote
            elif tag.find('a'):
                tag.decompose()

        # Remove elements with footnote-related classes
        for class_name in self.FOOTNOTE_CLASSES:
            for tag in soup.find_all(class_=re.compile(class_name, re.I)):
                tag.decompose()

        # Remove elements with footnote-related roles
        for role in self.FOOTNOTE_ROLES:
            for tag in soup.find_all(attrs={'role': role}):
                tag.decompose()
            for tag in soup.find_all(attrs={'epub:type': re.compile(role, re.I)}):
                tag.decompose()

        # Remove links that point to footnotes/endnotes
        for a in soup.find_all('a'):
            href = a.get('href', '')
            # Common footnote link patterns
            if re.search(r'#(fn|note|endnote|footnote|ref)', href, re.I):
                # Replace with just the text content (without the link)
                a.replace_with(a.get_text())
            # Links that contain only numbers are likely references
            elif a.get_text(strip=True).isdigit():
                a.decompose()

        # Remove standalone reference numbers in text
        # Pattern: numbers at end of sentences that look like references
        for text_node in soup.find_all(string=True):
            if text_node.parent.name not in ['script', 'style']:
                # Remove trailing reference numbers like "word.1" or "word1"
                cleaned = re.sub(r'(?<=[a-zA-Z.!?])(\d{1,3})(?=\s|$|[.!?])', '', str(text_node))
                if cleaned != str(text_node):
                    text_node.replace_with(cleaned)

        return soup

    # =========================================================================
    # PASS 2: Extract & Segment Sentences
    # =========================================================================

    def _normalize_with_index(self, text: str):
        """
        Collapse whitespace while preserving a mapping from normalized index
        back to the original text index.
        """
        normalized_chars = []
        norm_to_raw = []
        previous_was_space = False

        for raw_idx, char in enumerate(text):
            if char.isspace():
                if not previous_was_space:
                    normalized_chars.append(' ')
                    norm_to_raw.append(raw_idx)
                    previous_was_space = True
            else:
                normalized_chars.append(char)
                norm_to_raw.append(raw_idx)
                previous_was_space = False

        return ''.join(normalized_chars), norm_to_raw

    def _segment_text_with_offsets(self, text: str) -> list:
        """
        Use PySBD for accurate sentence segmentation and return spans with
        offsets mapped to the original raw text.
        """
        if not text or not text.strip():
            return []

        normalized_text, norm_to_raw = self._normalize_with_index(text)
        if not normalized_text.strip():
            return []

        segments = self.segmenter.segment(normalized_text)
        spans = []
        cursor = 0

        for segment in segments:
            if not segment:
                continue

            start = normalized_text.find(segment, cursor)
            if start == -1:
                start = normalized_text.find(segment)
                if start == -1:
                    continue

            end = start + len(segment)
            cursor = end

            # Trim leading/trailing whitespace from boundaries
            trimmed_start = start
            trimmed_end = end
            while trimmed_start < trimmed_end and normalized_text[trimmed_start].isspace():
                trimmed_start += 1
            while trimmed_end > trimmed_start and normalized_text[trimmed_end - 1].isspace():
                trimmed_end -= 1

            if trimmed_start >= trimmed_end:
                continue

            raw_start = norm_to_raw[trimmed_start]
            raw_end = norm_to_raw[trimmed_end - 1] + 1
            sentence_text = ' '.join(text[raw_start:raw_end].split())

            if len(sentence_text) < 3 or not any(c.isalpha() for c in sentence_text):
                continue

            spans.append({
                'text': sentence_text,
                'start': raw_start,
                'end': raw_end
            })

        return spans

    def _process_chapter(self, item):
        """Process a single chapter through both passes."""
        # Parse HTML
        soup = BeautifulSoup(item.get_content(), 'html.parser')

        # Extract chapter title
        chapter_title = self._extract_chapter_title(soup, item)

        # Process images
        self._process_chapter_images(soup)

        # PASS 1: Clean HTML
        soup = self._clean_html(soup)

        # PASS 2: Extract and segment sentences
        container = soup.body if soup.body else soup
        chapter_sentences, annotated_html = self._extract_and_annotate(container)

        # Generate chapter ID
        cid = hashlib.md5(chapter_title.encode('utf-8')).hexdigest()

        return {
            'id': cid,
            'title': chapter_title,
            'html': str(annotated_html),
            'sentences': chapter_sentences
        }

    def _extract_chapter_title(self, soup: BeautifulSoup, item) -> str:
        """Extract chapter title from heading or filename."""
        for tag in ['h1', 'h2', 'h3', 'title']:
            h = soup.find(tag)
            if h:
                return h.get_text(strip=True)

        # Fallback to filename
        name = os.path.basename(item.get_name())
        name = os.path.splitext(name)[0].replace('_', ' ').replace('-', ' ')
        return ' '.join(w.capitalize() for w in name.split())

    def _extract_and_annotate(self, root):
        """
        Extract sentences from block elements and annotate HTML with sentence spans.
        """
        chapter_sentences = []
        sentence_index = 0

        soup = BeautifulSoup(str(root), 'html.parser')

        for block in soup.find_all(self.BLOCK_TAGS):
            # Collect all text nodes in this block
            text_nodes = []
            for n in block.descendants:
                if isinstance(n, NavigableString) and n.parent.name not in ('script', 'style'):
                    text_nodes.append(n)

            if not text_nodes:
                continue

            # Build full text and mapping
            parts, map_ranges = [], []
            cursor = 0
            for n in text_nodes:
                t = str(n)
                parts.append(t)
                start, end = cursor, cursor + len(t)
                map_ranges.append((n, start, end))
                cursor = end

            full_text = ''.join(parts)
            if not full_text.strip():
                continue

            # PASS 2: Use PySBD for sentence segmentation
            spans = self._segment_text_with_offsets(full_text)
            if not spans:
                continue

            # Generate IDs and collect sentences
            for s in spans:
                sid = self._generate_sentence_id(s['text'], sentence_index)
                s['id'] = sid
                s['sentence_index'] = sentence_index
                chapter_sentences.append({
                    'id': sid,
                    'text': s['text'],
                    'sentence_index': sentence_index
                })
                sentence_index += 1

            # Annotate HTML with sentence spans
            self._annotate_block(soup, map_ranges, spans)

        return chapter_sentences, soup

    def _annotate_block(self, soup, map_ranges, spans):
        """Wrap text nodes with sentence span tags."""
        for node, ns, ne in map_ranges:
            original = str(node)
            if not original:
                continue

            overlapping = [s for s in spans if not (s['end'] <= ns or s['start'] >= ne)]
            if not overlapping:
                continue

            overlapping.sort(key=lambda x: x['start'])

            new_nodes = []
            local_cursor = 0
            nlen = len(original)

            for s in overlapping:
                ls = max(s['start'], ns) - ns
                le = min(s['end'], ne) - ns

                if ls > local_cursor:
                    before = original[local_cursor:ls]
                    if before:
                        new_nodes.append(before)

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
                if after:
                    new_nodes.append(after)

            for new_node in reversed(new_nodes):
                node.insert_after(new_node)
            node.extract()

    def _generate_sentence_id(self, text: str, index: int) -> str:
        """Generate unique ID for a sentence."""
        key = f"{index}:{text.strip()[:100]}"
        return hashlib.md5(key.encode('utf-8')).hexdigest()

    def _process_chapter_images(self, soup):
        """Update image src attributes to use base64 data URIs."""
        for img in soup.find_all('img'):
            src = img.get('src')
            if not src:
                continue
            clean = src.replace('../', '')
            for path, data in self.images.items():
                base = os.path.basename(path)
                if clean in path or path.endswith(clean) or base == os.path.basename(src):
                    img['src'] = data
                    img['style'] = 'max-width:100%;height:auto;display:block;margin:10px auto;'
                    break
