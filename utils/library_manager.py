# utils/library_manager.py
"""
Library manager for tracking opened books and their progress.
Stores book metadata, reading progress, and audio generation status.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
import threading


class LibraryManager:
    """Manages the user's book library with progress tracking."""

    def __init__(self, library_path: str = 'static/library'):
        self.library_path = Path(library_path)
        self.library_file = self.library_path / 'library.json'
        self.covers_path = self.library_path / 'covers'
        self._lock = threading.Lock()

        # Ensure directories exist
        self.library_path.mkdir(parents=True, exist_ok=True)
        self.covers_path.mkdir(parents=True, exist_ok=True)

        self.library = self._load_library()

    def _load_library(self) -> Dict:
        """Load library from JSON file."""
        if self.library_file.exists():
            try:
                with open(self.library_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading library: {e}")
        return {'books': {}}

    def _save_library(self):
        """Save library to JSON file."""
        with self._lock:
            try:
                with open(self.library_file, 'w') as f:
                    json.dump(self.library, f, indent=2)
            except Exception as e:
                print(f"Error saving library: {e}")

    def add_book(self, book_id: str, title: str, author: str = '',
                 cover_data: Optional[str] = None, total_chapters: int = 0,
                 total_sentences: int = 0, file_path: str = None) -> Dict:
        """Add or update a book in the library."""
        now = datetime.now().isoformat()

        if book_id in self.library['books']:
            # Update existing book
            book = self.library['books'][book_id]
            book['last_opened'] = now
            book['open_count'] = book.get('open_count', 0) + 1
            # Update metadata if provided
            if title:
                book['title'] = title
            if author:
                book['author'] = author
            if total_chapters:
                book['total_chapters'] = total_chapters
            if total_sentences:
                book['total_sentences'] = total_sentences
            if file_path:
                book['file_path'] = file_path
        else:
            # Add new book
            self.library['books'][book_id] = {
                'book_id': book_id,
                'title': title,
                'author': author,
                'file_path': file_path,
                'added_date': now,
                'last_opened': now,
                'open_count': 1,
                'total_chapters': total_chapters,
                'total_sentences': total_sentences,
                'reading_progress': {
                    'current_chapter': 0,
                    'current_sentence_index': 0,
                    'percentage': 0
                },
                'audio_progress': {
                    'ready_count': 0,
                    'generating_count': 0,
                    'percentage': 0
                }
            }

        # Save cover image if provided
        if cover_data:
            self._save_cover(book_id, cover_data)

        self._save_library()
        return self.library['books'][book_id]

    def _save_cover(self, book_id: str, cover_data: str):
        """Save cover image (base64 data URI or path)."""
        cover_file = self.covers_path / f"{book_id}.jpg"
        try:
            # If it's a data URI, extract and save
            if cover_data.startswith('data:image'):
                import base64
                # Extract base64 data
                header, data = cover_data.split(',', 1)
                img_data = base64.b64decode(data)
                with open(cover_file, 'wb') as f:
                    f.write(img_data)
                self.library['books'][book_id]['cover_path'] = str(cover_file)
        except Exception as e:
            print(f"Error saving cover: {e}")

    def update_reading_progress(self, book_id: str, current_chapter: int,
                                 current_sentence_index: int = 0,
                                 total_chapters: int = None):
        """Update reading progress for a book."""
        if book_id not in self.library['books']:
            return

        book = self.library['books'][book_id]
        progress = book['reading_progress']
        progress['current_chapter'] = current_chapter
        progress['current_sentence_index'] = current_sentence_index

        # Calculate percentage
        total = total_chapters or book.get('total_chapters', 1)
        if total > 0:
            progress['percentage'] = round((current_chapter / total) * 100, 1)

        book['last_opened'] = datetime.now().isoformat()
        self._save_library()

    def update_audio_progress(self, book_id: str, ready_count: int,
                               total_sentences: int):
        """Update audio generation progress for a book."""
        if book_id not in self.library['books']:
            return

        book = self.library['books'][book_id]
        progress = book['audio_progress']
        progress['ready_count'] = ready_count
        progress['total_sentences'] = total_sentences

        if total_sentences > 0:
            progress['percentage'] = round((ready_count / total_sentences) * 100, 1)

        self._save_library()

    def get_book(self, book_id: str) -> Optional[Dict]:
        """Get a book's metadata and progress."""
        return self.library['books'].get(book_id)

    def get_all_books(self) -> List[Dict]:
        """Get all books sorted by last opened date."""
        books = list(self.library['books'].values())
        # Sort by last opened (most recent first)
        books.sort(key=lambda b: b.get('last_opened', ''), reverse=True)
        return books

    def remove_book(self, book_id: str):
        """Remove a book from the library."""
        if book_id in self.library['books']:
            del self.library['books'][book_id]
            # Remove cover if exists
            cover_file = self.covers_path / f"{book_id}.jpg"
            if cover_file.exists():
                cover_file.unlink()
            self._save_library()

    def get_cover_url(self, book_id: str) -> Optional[str]:
        """Get the URL for a book's cover image."""
        book = self.library['books'].get(book_id)
        if book and 'cover_path' in book:
            return f"/library/cover/{book_id}"
        return None


# Global singleton
_library_manager = None
_manager_lock = threading.Lock()


def get_library_manager() -> LibraryManager:
    """Get singleton library manager instance."""
    global _library_manager
    if _library_manager is None:
        with _manager_lock:
            if _library_manager is None:
                _library_manager = LibraryManager()
    return _library_manager
