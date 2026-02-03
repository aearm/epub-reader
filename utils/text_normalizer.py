# utils/text_normalizer.py
"""
Text normalization for TTS input.
Converts abbreviations, numbers, and special text to spoken form.
"""
import re
from num2words import num2words


class TextNormalizer:
    """Normalize text for TTS pronunciation."""

    # Title abbreviations
    TITLE_ABBREVS = {
        'Dr.': 'Doctor',
        'Mr.': 'Mister',
        'Mrs.': 'Missus',
        'Ms.': 'Miss',
        'Prof.': 'Professor',
        'Rev.': 'Reverend',
        'Sr.': 'Senior',
        'Jr.': 'Junior',
        'Hon.': 'Honorable',
        'Gov.': 'Governor',
        'Gen.': 'General',
        'Col.': 'Colonel',
        'Lt.': 'Lieutenant',
        'Sgt.': 'Sergeant',
        'Capt.': 'Captain',
        'Adm.': 'Admiral',
        'Rep.': 'Representative',
        'Sen.': 'Senator',
        'Pres.': 'President',
        'Supt.': 'Superintendent',
        'Fr.': 'Father',
        'Br.': 'Brother',
        'Sr.': 'Sister',
    }

    # Common abbreviations
    COMMON_ABBREVS = {
        'vs.': 'versus',
        'etc.': 'etcetera',
        'e.g.': 'for example',
        'i.e.': 'that is',
        'approx.': 'approximately',
        'dept.': 'department',
        'govt.': 'government',
        'no.': 'number',
        'vol.': 'volume',
        'pg.': 'page',
        'ch.': 'chapter',
        'fig.': 'figure',
        'inc.': 'incorporated',
        'corp.': 'corporation',
        'ltd.': 'limited',
        'co.': 'company',
        'est.': 'established',
        'max.': 'maximum',
        'min.': 'minimum',
        'avg.': 'average',
        'qty.': 'quantity',
        'approx.': 'approximately',
        'misc.': 'miscellaneous',
        'ref.': 'reference',
        'ex.': 'example',
        'assn.': 'association',
        'dept.': 'department',
        'intl.': 'international',
        'natl.': 'national',
    }

    # Location abbreviations
    LOCATION_ABBREVS = {
        'St.': 'Street',
        'Ave.': 'Avenue',
        'Blvd.': 'Boulevard',
        'Rd.': 'Road',
        'Dr.': 'Drive',  # Note: context-dependent, handled separately
        'Ln.': 'Lane',
        'Ct.': 'Court',
        'Pl.': 'Place',
        'Apt.': 'Apartment',
        'Ste.': 'Suite',
        'Bldg.': 'Building',
        'Mt.': 'Mount',
        'Ft.': 'Fort',
    }

    # Month abbreviations
    MONTH_ABBREVS = {
        'Jan.': 'January',
        'Feb.': 'February',
        'Mar.': 'March',
        'Apr.': 'April',
        'Jun.': 'June',
        'Jul.': 'July',
        'Aug.': 'August',
        'Sep.': 'September',
        'Sept.': 'September',
        'Oct.': 'October',
        'Nov.': 'November',
        'Dec.': 'December',
    }

    # Day abbreviations
    DAY_ABBREVS = {
        'Mon.': 'Monday',
        'Tue.': 'Tuesday',
        'Tues.': 'Tuesday',
        'Wed.': 'Wednesday',
        'Thu.': 'Thursday',
        'Thur.': 'Thursday',
        'Thurs.': 'Thursday',
        'Fri.': 'Friday',
        'Sat.': 'Saturday',
        'Sun.': 'Sunday',
    }

    def __init__(self):
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile regex patterns for efficiency."""
        # Footnote/reference patterns - numbers that shouldn't be read aloud

        # Superscript Unicode characters: ¹²³⁴⁵⁶⁷⁸⁹⁰
        self.superscript_pattern = re.compile(r'[\u00B9\u00B2\u00B3\u2070-\u2079]+')

        # Bracketed references: [1], [2], [12]
        self.bracket_ref_pattern = re.compile(r'\[\d{1,3}\]')

        # Parenthetical references: (1), (2)
        self.paren_ref_pattern = re.compile(r'\(\d{1,3}\)')

        # Footnote numbers after punctuation at end: "word.3" or "word,3" or "word3"
        # Match 1-3 digits that appear:
        # - After a letter followed by punctuation: reading.3
        # - After a letter at word end: hemisphere2
        # - At the very end of text
        self.footnote_after_punct = re.compile(r'([.!?;:,])(\d{1,3})(?=\s|$|[)\]\"\'])')
        self.footnote_after_letter = re.compile(r'([a-zA-Z])(\d{1,3})(?=\s|$|[.!?;:,)\]\"\'])')

        # Currency pattern: $1,234.56
        self.currency_pattern = re.compile(
            r'([\$\u00A3\u20AC\u00A5])(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)'
        )

        # Number with commas: 1,234,567
        self.number_with_commas_pattern = re.compile(
            r'\b(\d{1,3}(?:,\d{3})+)\b'
        )

        # Large numbers without commas (5+ digits)
        self.large_number_pattern = re.compile(
            r'\b(\d{5,})\b'
        )

        # Ordinal numbers: 1st, 2nd, 3rd, 4th
        self.ordinal_pattern = re.compile(
            r'\b(\d+)(st|nd|rd|th)\b', re.IGNORECASE
        )

        # Time: 3:30 PM or 15:30
        self.time_pattern = re.compile(
            r'\b(\d{1,2}):(\d{2})(?:\s*(AM|PM|am|pm|a\.m\.|p\.m\.))?\b'
        )

        # Percentages: 50%
        self.percent_pattern = re.compile(
            r'\b(\d+(?:\.\d+)?)\s*%'
        )

        # Fractions: 1/2, 3/4
        self.fraction_pattern = re.compile(
            r'\b(\d+)/(\d+)\b'
        )

        # Roman numerals (when clearly used as numerals, e.g., "Chapter IV")
        self.roman_numeral_pattern = re.compile(
            r'\b(Chapter|Part|Book|Volume|Act|Scene|Section)\s+((?:I{1,3}|IV|V|VI{0,3}|IX|X{1,3}|L|C)+)\b',
            re.IGNORECASE
        )

        # Phone numbers: (123) 456-7890 or 123-456-7890
        self.phone_pattern = re.compile(
            r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        )

        # Email addresses (simple pattern)
        self.email_pattern = re.compile(
            r'\b[\w.+-]+@[\w.-]+\.\w{2,}\b'
        )

        # URLs
        self.url_pattern = re.compile(
            r'https?://[^\s<>"{}|\\^`\[\]]+',
            re.IGNORECASE
        )

        # Years: 1990, 2024
        self.year_pattern = re.compile(
            r'\b(1[89]\d{2}|20[0-2]\d)\b'
        )

        # Decades: 1990s, 80s
        self.decade_pattern = re.compile(
            r"\b(\d{2,4})'?s\b"
        )

    def normalize(self, text: str) -> str:
        """
        Apply all normalizations to text.
        Order matters - more specific patterns first.
        """
        if not text or not text.strip():
            return text

        # Step 0: Remove footnote/reference numbers (MUST be first!)
        text = self._remove_references(text)

        # Step 1: Handle URLs (remove or describe)
        text = self._normalize_urls(text)

        # Step 2: Handle email addresses
        text = self._normalize_emails(text)

        # Step 3: Expand title abbreviations (Dr., Mr., etc.)
        text = self._expand_titles(text)

        # Step 4: Expand month/day abbreviations
        text = self._expand_dates(text)

        # Step 5: Expand common abbreviations
        text = self._expand_abbreviations(text)

        # Step 6: Handle location abbreviations (careful with Dr./Drive)
        text = self._expand_locations(text)

        # Step 7: Normalize currency
        text = self._normalize_currency(text)

        # Step 8: Normalize percentages
        text = self._normalize_percentages(text)

        # Step 9: Normalize fractions
        text = self._normalize_fractions(text)

        # Step 10: Normalize time
        text = self._normalize_time(text)

        # Step 11: Normalize ordinals
        text = self._normalize_ordinals(text)

        # Step 12: Normalize decades (1990s -> nineteen nineties)
        text = self._normalize_decades(text)

        # Step 13: Normalize years
        text = self._normalize_years(text)

        # Step 14: Normalize numbers with commas
        text = self._normalize_numbers_with_commas(text)

        # Step 15: Normalize large numbers
        text = self._normalize_large_numbers(text)

        # Step 16: Handle special characters
        text = self._normalize_special_chars(text)

        # Step 17: Clean up extra whitespace
        text = ' '.join(text.split())

        return text

    def _remove_references(self, text: str) -> str:
        """
        Remove footnote and reference numbers that shouldn't be read aloud.
        Examples: word² -> word, sentence.[1] -> sentence., reading.3 -> reading.
        """
        # Remove superscript Unicode characters: ¹²³ etc.
        text = self.superscript_pattern.sub('', text)

        # Remove bracketed references [1], [2], [12]
        text = self.bracket_ref_pattern.sub('', text)

        # Remove parenthetical single-number references (1), (2)
        text = self.paren_ref_pattern.sub('', text)

        # Remove footnote numbers after punctuation: "reading.3" -> "reading."
        # Keep the punctuation, remove the number
        text = self.footnote_after_punct.sub(r'\1', text)

        # Remove footnote numbers after letters: "hemisphere2" -> "hemisphere"
        # Keep the letter, remove the number
        text = self.footnote_after_letter.sub(r'\1', text)

        return text

    def _expand_titles(self, text: str) -> str:
        """Expand title abbreviations."""
        for abbrev, expansion in self.TITLE_ABBREVS.items():
            # Use word boundary to avoid partial matches
            pattern = re.compile(r'\b' + re.escape(abbrev), re.IGNORECASE)
            text = pattern.sub(expansion, text)
        return text

    def _expand_dates(self, text: str) -> str:
        """Expand month and day abbreviations."""
        for abbrev, expansion in self.MONTH_ABBREVS.items():
            pattern = re.compile(r'\b' + re.escape(abbrev), re.IGNORECASE)
            text = pattern.sub(expansion, text)
        for abbrev, expansion in self.DAY_ABBREVS.items():
            pattern = re.compile(r'\b' + re.escape(abbrev), re.IGNORECASE)
            text = pattern.sub(expansion, text)
        return text

    def _expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations."""
        for abbrev, expansion in self.COMMON_ABBREVS.items():
            pattern = re.compile(r'\b' + re.escape(abbrev), re.IGNORECASE)
            text = pattern.sub(expansion, text)
        return text

    def _expand_locations(self, text: str) -> str:
        """Expand location abbreviations."""
        # Skip Dr. as it's handled in titles (context: "Dr. Smith" vs "123 Oak Dr.")
        for abbrev, expansion in self.LOCATION_ABBREVS.items():
            if abbrev == 'Dr.':
                # Only expand Dr. when preceded by a number (address)
                text = re.sub(r'(\d+\s+\w+)\s+Dr\.', r'\1 Drive', text)
            else:
                pattern = re.compile(r'\b' + re.escape(abbrev), re.IGNORECASE)
                text = pattern.sub(expansion, text)
        return text

    def _normalize_currency(self, text: str) -> str:
        """Convert currency to spoken form."""
        currency_names = {
            '$': ('dollar', 'dollars', 'cent', 'cents'),
            '\u00A3': ('pound', 'pounds', 'penny', 'pence'),
            '\u20AC': ('euro', 'euros', 'cent', 'cents'),
            '\u00A5': ('yen', 'yen', '', ''),
        }

        def replace_currency(match):
            symbol = match.group(1)
            amount_str = match.group(2).replace(',', '')
            names = currency_names.get(symbol, ('dollar', 'dollars', 'cent', 'cents'))

            try:
                if '.' in amount_str:
                    dollars, cents = amount_str.split('.')
                    dollars = int(dollars)
                    cents = int(cents)

                    dollar_word = num2words(dollars)
                    dollar_unit = names[0] if dollars == 1 else names[1]

                    if cents > 0 and names[2]:  # Has cent units
                        cent_word = num2words(cents)
                        cent_unit = names[2] if cents == 1 else names[3]
                        return f"{dollar_word} {dollar_unit} and {cent_word} {cent_unit}"
                    return f"{dollar_word} {dollar_unit}"
                else:
                    amount = int(amount_str)
                    amount_word = num2words(amount)
                    unit = names[0] if amount == 1 else names[1]
                    return f"{amount_word} {unit}"
            except (ValueError, OverflowError):
                return match.group(0)  # Return original if conversion fails

        return self.currency_pattern.sub(replace_currency, text)

    def _normalize_percentages(self, text: str) -> str:
        """Convert percentages to spoken form."""
        def replace_percent(match):
            number = match.group(1)
            try:
                if '.' in number:
                    return f"{num2words(float(number))} percent"
                return f"{num2words(int(number))} percent"
            except (ValueError, OverflowError):
                return match.group(0)

        return self.percent_pattern.sub(replace_percent, text)

    def _normalize_fractions(self, text: str) -> str:
        """Convert fractions to spoken form."""
        def replace_fraction(match):
            numerator = int(match.group(1))
            denominator = int(match.group(2))

            # Common fractions
            if numerator == 1 and denominator == 2:
                return "one half"
            elif numerator == 1 and denominator == 4:
                return "one quarter"
            elif numerator == 3 and denominator == 4:
                return "three quarters"
            elif numerator == 1 and denominator == 3:
                return "one third"
            elif numerator == 2 and denominator == 3:
                return "two thirds"
            else:
                try:
                    num_word = num2words(numerator)
                    denom_word = num2words(denominator, to='ordinal')
                    if numerator > 1:
                        denom_word += 's'
                    return f"{num_word} {denom_word}"
                except (ValueError, OverflowError):
                    return match.group(0)

        return self.fraction_pattern.sub(replace_fraction, text)

    def _normalize_time(self, text: str) -> str:
        """Convert time to spoken form."""
        def replace_time(match):
            hour = int(match.group(1))
            minute = int(match.group(2))
            period = match.group(3) or ''

            if period:
                period = period.replace('.', '').upper()
                if period == 'AM':
                    period = ' AM'
                elif period == 'PM':
                    period = ' PM'

            try:
                hour_word = num2words(hour)

                if minute == 0:
                    if period:
                        return f"{hour_word}{period}"
                    return f"{hour_word} o'clock"
                elif minute < 10:
                    minute_word = f"oh {num2words(minute)}"
                else:
                    minute_word = num2words(minute)

                return f"{hour_word} {minute_word}{period}"
            except (ValueError, OverflowError):
                return match.group(0)

        return self.time_pattern.sub(replace_time, text)

    def _normalize_ordinals(self, text: str) -> str:
        """Convert ordinals to spoken form."""
        def replace_ordinal(match):
            try:
                number = int(match.group(1))
                return num2words(number, to='ordinal')
            except (ValueError, OverflowError):
                return match.group(0)

        return self.ordinal_pattern.sub(replace_ordinal, text)

    def _normalize_decades(self, text: str) -> str:
        """Convert decades to spoken form (1990s -> nineteen nineties)."""
        def replace_decade(match):
            decade = match.group(1)
            try:
                if len(decade) == 4:
                    # 1990s -> nineteen nineties
                    century = int(decade[:2])
                    tens = int(decade[2:])
                    century_word = num2words(century)
                    if tens == 0:
                        return f"the {century_word} hundreds"
                    tens_word = num2words(tens * 10)
                    return f"the {century_word} {tens_word}s"
                else:
                    # 90s -> nineties
                    tens = int(decade)
                    return f"the {num2words(tens)}s"
            except (ValueError, OverflowError):
                return match.group(0)

        return self.decade_pattern.sub(replace_decade, text)

    def _normalize_years(self, text: str) -> str:
        """Convert years to spoken form (2024 -> twenty twenty-four)."""
        def replace_year(match):
            year = int(match.group(1))
            try:
                if year >= 2000 and year < 2010:
                    return num2words(year)  # "two thousand five"
                elif year >= 2010:
                    # 2024 -> twenty twenty-four
                    century = year // 100
                    rest = year % 100
                    if rest == 0:
                        return num2words(century) + " hundred"
                    return f"{num2words(century)} {num2words(rest)}"
                else:
                    # 1990 -> nineteen ninety
                    century = year // 100
                    rest = year % 100
                    if rest == 0:
                        return num2words(century) + " hundred"
                    return f"{num2words(century)} {num2words(rest)}"
            except (ValueError, OverflowError):
                return match.group(0)

        return self.year_pattern.sub(replace_year, text)

    def _normalize_numbers_with_commas(self, text: str) -> str:
        """Convert numbers with commas to spoken form."""
        def replace_number(match):
            try:
                number = int(match.group(1).replace(',', ''))
                return num2words(number)
            except (ValueError, OverflowError):
                return match.group(0)

        return self.number_with_commas_pattern.sub(replace_number, text)

    def _normalize_large_numbers(self, text: str) -> str:
        """Convert large numbers (5+ digits) to spoken form."""
        def replace_number(match):
            try:
                number = int(match.group(1))
                return num2words(number)
            except (ValueError, OverflowError):
                return match.group(0)

        return self.large_number_pattern.sub(replace_number, text)

    def _normalize_urls(self, text: str) -> str:
        """Handle URLs - replace with description."""
        def replace_url(match):
            return " link "

        return self.url_pattern.sub(replace_url, text)

    def _normalize_emails(self, text: str) -> str:
        """Handle email addresses - spell out @ as 'at'."""
        def replace_email(match):
            email = match.group(0)
            return email.replace('@', ' at ').replace('.', ' dot ')

        return self.email_pattern.sub(replace_email, text)

    def _normalize_special_chars(self, text: str) -> str:
        """Replace special characters with spoken equivalents."""
        replacements = [
            ('&', ' and '),
            ('@', ' at '),
            ('#', ' number '),
            ('+', ' plus '),
            ('=', ' equals '),
            ('<', ' less than '),
            ('>', ' greater than '),
            ('...', ', '),
            ('\u2026', ', '),  # Ellipsis character
            ('\u2014', ', '),  # Em dash
            ('\u2013', ', '),  # En dash
            ('--', ', '),
            ('/', ' slash '),
            ('*', ' '),
            ('_', ' '),
            ('"', ''),
            ("'", "'"),  # Keep apostrophes
            ('\u201C', ''),  # Left double quote
            ('\u201D', ''),  # Right double quote
            ('\u2018', "'"),  # Left single quote
            ('\u2019', "'"),  # Right single quote (apostrophe)
        ]

        for char, replacement in replacements:
            text = text.replace(char, replacement)

        return text


# Global normalizer instance
_normalizer = None


def get_normalizer() -> TextNormalizer:
    """Get singleton normalizer instance."""
    global _normalizer
    if _normalizer is None:
        _normalizer = TextNormalizer()
    return _normalizer


def normalize_text(text: str) -> str:
    """Convenience function to normalize text for TTS."""
    return get_normalizer().normalize(text)
