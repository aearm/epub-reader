# utils/reading_state.py
"""
Reading state persistence for EPUB reader.
Stores last reading position per book.
"""
import json
import os
from datetime import datetime
from typing import Dict, Optional, List


class ReadingStateManager:
    """Manage reading state persistence."""

    def __init__(self, state_dir: str = 'static/state'):
        self.state_dir = state_dir
        os.makedirs(state_dir, exist_ok=True)

    def _get_state_path(self, book_id: str) -> str:
        """Get path to state file for a book."""
        return os.path.join(self.state_dir, f"{book_id}.json")

    def save_position(self, book_id: str, position: Dict) -> bool:
        """
        Save reading position for a book.

        position: {
            'chapter_index': int,
            'page_index': int,
            'sentence_id': str,
            'sentence_index': int,
            'scroll_position': float,  # 0-1 percentage
        }
        """
        try:
            state_path = self._get_state_path(book_id)

            # Load existing state or create new
            state = self.load_state(book_id) or {}

            # Update position with timestamp
            position['timestamp'] = datetime.now().isoformat()
            state['position'] = position
            state['last_accessed'] = datetime.now().isoformat()

            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)

            return True
        except Exception as e:
            print(f"Error saving position for {book_id}: {e}")
            return False

    def load_position(self, book_id: str) -> Optional[Dict]:
        """Load reading position for a book."""
        state = self.load_state(book_id)
        return state.get('position') if state else None

    def load_state(self, book_id: str) -> Optional[Dict]:
        """Load full state for a book."""
        state_path = self._get_state_path(book_id)

        if not os.path.exists(state_path):
            return None

        try:
            with open(state_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading state for {book_id}: {e}")
            return None

    def save_book_metadata(self, book_id: str, title: str, total_chapters: int) -> bool:
        """Save book metadata for recent books list."""
        try:
            state_path = self._get_state_path(book_id)
            state = self.load_state(book_id) or {}

            state['book_id'] = book_id
            state['title'] = title
            state['total_chapters'] = total_chapters
            state['last_accessed'] = datetime.now().isoformat()

            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)

            return True
        except Exception as e:
            print(f"Error saving book metadata for {book_id}: {e}")
            return False

    def save_audio_status(self, book_id: str, audio_status: Dict) -> bool:
        """Save audio generation status summary for a book."""
        try:
            state_path = self._get_state_path(book_id)
            state = self.load_state(book_id) or {}

            # Only save summary counts, not full status dict
            ready_count = sum(1 for s in audio_status.values() if s == 'ready')
            total_count = len(audio_status)

            state['audio_progress'] = {
                'ready': ready_count,
                'total': total_count,
                'updated': datetime.now().isoformat()
            }

            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)

            return True
        except Exception as e:
            print(f"Error saving audio status for {book_id}: {e}")
            return False

    def get_recent_books(self, limit: int = 10) -> List[Dict]:
        """Get list of recently accessed books."""
        books = []

        if not os.path.exists(self.state_dir):
            return books

        for filename in os.listdir(self.state_dir):
            if filename.endswith('.json'):
                book_id = filename[:-5]  # Remove .json
                state = self.load_state(book_id)
                if state:
                    books.append({
                        'book_id': book_id,
                        'title': state.get('title', 'Unknown'),
                        'last_accessed': state.get('last_accessed'),
                        'position': state.get('position'),
                        'audio_progress': state.get('audio_progress'),
                        'total_chapters': state.get('total_chapters', 0)
                    })

        # Sort by last accessed (most recent first)
        books.sort(key=lambda x: x.get('last_accessed', ''), reverse=True)
        return books[:limit]

    def delete_book_state(self, book_id: str) -> bool:
        """Delete saved state for a book."""
        try:
            state_path = self._get_state_path(book_id)
            if os.path.exists(state_path):
                os.remove(state_path)
            return True
        except Exception as e:
            print(f"Error deleting state for {book_id}: {e}")
            return False


# Global manager instance
_state_manager = None


def get_state_manager() -> ReadingStateManager:
    """Get singleton state manager."""
    global _state_manager
    if _state_manager is None:
        _state_manager = ReadingStateManager()
    return _state_manager
