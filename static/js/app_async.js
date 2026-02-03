// EPUB Reader with AI Chat
class ProgressiveEPUBReader {
    constructor() {
        this.socket = null;
        this.currentBook = null;
        this.currentChapter = 0;
        this.currentSentence = null;
        this.audioStatus = {};
        this.audioElements = {};
        this.sentences = [];
        this.currentAudio = null;
        this.isPlaying = false;
        this.isTransitioning = false;
        this.autoPlay = false;
        this.playbackRate = 1.0;
        this.audioReadyCount = 0;
        this.totalSentences = 0;
        this.notes = {};
        this.selectedSentenceId = null;
        this.selectedText = null;
        this.pendingNoteText = null;
        this.pendingChatContext = null;
        this.ollamaConnected = false;
        this.ollamaModels = [];
        this.currentSidebarView = 'chapters'; // 'chapters' or 'notes'
        this.lastReadPosition = null; // Separate from note navigation

        // Focus mode state
        this.focusModeActive = false;
        this.focusSentences = [];
        this.focusCurrentIndex = 0;
        this.focusAutoplay = false;
        this.focusIsPlaying = false;
        this.focusReturnSentenceId = null; // To return to when exiting

        // Settings
        this.currentTheme = localStorage.getItem('theme') || 'light';
        this.bionicEnabled = localStorage.getItem('bionic') === 'true';

        // Translation
        this.motherTongue = localStorage.getItem('motherTongue') || '';
        this.translationCache = {};
        this.hoverTimeout = null;

        // Session & Stats
        this.sessionActive = false;
        this.sessionTimeLeft = 0;
        this.sessionInterval = null;
        this.stats = this.loadStats();

        this.initializeElements();
        this.attachEventListeners();
        this.initializeWebSocket();
        this.initializeOllama();
        this.startPositionTracking();
        this.loadLibrary();
    }

    initializeElements() {
        // File upload
        this.fileInput = document.getElementById('fileInput');
        this.uploadBtn = document.getElementById('uploadBtn');
        this.clearBtn = document.getElementById('clearBtn');

        // Loading overlay
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.loadingText = document.getElementById('loadingText');

        // Header
        this.headerTitle = document.getElementById('headerTitle');
        this.toggleChaptersBtn = document.getElementById('toggleChapters');
        this.toggleNotesBtn = document.getElementById('toggleNotes');
        this.toggleFocusBtn = document.getElementById('toggleFocus');
        this.toggleSettingsBtn = document.getElementById('toggleSettings');
        this.toggleChatBtn = document.getElementById('toggleChat');

        // Main containers
        this.welcomeScreen = document.getElementById('welcomeScreen');
        this.readerContainer = document.getElementById('readerContainer');

        // Sidebars
        this.leftSidebar = document.getElementById('leftSidebar');
        this.chatSidebar = document.getElementById('chatSidebar');

        // Sidebar views
        this.chaptersView = document.getElementById('chaptersView');
        this.notesView = document.getElementById('notesView');
        this.notesList = document.getElementById('notesList');
        this.notesCount = document.getElementById('notesCount');
        this.exportNotesBtn = document.getElementById('exportNotes');

        // Book content
        this.bookTitle = document.getElementById('bookTitle');
        this.chapterList = document.getElementById('chapterList');
        this.chapterContent = document.getElementById('chapterContent');

        // Audio progress
        this.audioReady = document.getElementById('audioReady');
        this.audioTotal = document.getElementById('audioTotal');
        this.audioProgressBar = document.getElementById('audioProgressBar');

        // Audio controls
        this.audioControls = document.getElementById('audioControls');
        this.playPauseBtn = document.getElementById('playPauseBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.prevBtn = document.getElementById('prevBtn');
        this.nextBtn = document.getElementById('nextBtn');
        this.speedSlider = document.getElementById('speedSlider');
        this.speedValue = document.getElementById('speedValue');
        this.autoplayCheckbox = document.getElementById('autoplayCheckbox');

        // Chat
        this.chatMessages = document.getElementById('chatMessages');
        this.chatInput = document.getElementById('chatInput');
        this.sendChatBtn = document.getElementById('sendChat');
        this.clearChatBtn = document.getElementById('clearChat');
        this.ollamaStatus = document.getElementById('ollamaStatus');
        this.ollamaStatusText = document.getElementById('ollamaStatusText');

        // Context menu
        this.contextMenu = document.getElementById('contextMenu');

        // Note modal
        this.noteModal = document.getElementById('noteModal');
        this.noteSelectedText = document.getElementById('noteSelectedText');
        this.noteInput = document.getElementById('noteInput');
        this.saveNoteBtn = document.getElementById('saveNote');

        // Focus mode
        this.focusMode = document.getElementById('focusMode');
        this.focusParagraph = document.getElementById('focusParagraph');
        this.focusProgressBar = document.getElementById('focusProgressBar');
        this.focusProgressText = document.getElementById('focusProgressText');
        this.focusPrevBtn = document.getElementById('focusPrev');
        this.focusNextBtn = document.getElementById('focusNext');
        this.focusPlayBtn = document.getElementById('focusPlay');
        this.focusPlayIcon = document.getElementById('focusPlayIcon');
        this.focusPauseIcon = document.getElementById('focusPauseIcon');
        this.focusAutoplayCheckbox = document.getElementById('focusAutoplay');
        this.exitFocusBtn = document.getElementById('exitFocus');

        // Settings panel
        this.settingsPanel = document.getElementById('settingsPanel');
        this.closeSettingsBtn = document.getElementById('closeSettings');
        this.bionicToggle = document.getElementById('bionicToggle');

        // Session timer
        this.sessionTimer = document.getElementById('sessionTimer');
        this.sessionTimeLeftEl = document.getElementById('sessionTimeLeft');

        // Stats
        this.statStreak = document.getElementById('statStreak');
        this.statToday = document.getElementById('statToday');
        this.statTotal = document.getElementById('statTotal');

        // Celebration
        this.celebration = document.getElementById('celebration');

        // Translation
        this.motherTongueSelect = document.getElementById('motherTongue');
        this.translationTooltip = document.getElementById('translationTooltip');

        // Library
        this.librarySection = document.getElementById('librarySection');
        this.libraryGrid = document.getElementById('libraryGrid');

        // Validate critical elements
        const criticalElements = {
            toggleSettingsBtn: this.toggleSettingsBtn,
            toggleChatBtn: this.toggleChatBtn,
            settingsPanel: this.settingsPanel,
            chatSidebar: this.chatSidebar
        };

        for (const [name, el] of Object.entries(criticalElements)) {
            if (!el) {
                console.error(`CRITICAL: Element '${name}' not found in DOM!`);
            }
        }
    }

    attachEventListeners() {
        // File upload
        this.uploadBtn.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', (e) => this.handleFileUpload(e));
        this.clearBtn.addEventListener('click', () => this.clearBook());

        // Sidebar toggles
        this.toggleChaptersBtn.addEventListener('click', () => this.showChaptersView());
        this.toggleNotesBtn.addEventListener('click', () => this.showNotesView());
        this.toggleChatBtn.addEventListener('click', () => this.toggleChat());
        this.exportNotesBtn.addEventListener('click', () => this.exportNotes());

        // Audio controls
        this.playPauseBtn.addEventListener('click', () => this.togglePlayPause());
        this.stopBtn.addEventListener('click', () => this.stopPlayback());
        this.prevBtn.addEventListener('click', () => this.playPreviousSentence());
        this.nextBtn.addEventListener('click', () => this.playNextSentence());
        this.speedSlider.addEventListener('input', (e) => this.updatePlaybackSpeed(e));
        this.autoplayCheckbox.addEventListener('change', (e) => this.toggleAutoplay(e));

        // Chat
        this.sendChatBtn.addEventListener('click', () => this.sendChatMessage());
        this.chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendChatMessage();
            }
        });
        this.clearChatBtn.addEventListener('click', () => this.clearChat());

        // Context menu
        document.addEventListener('contextmenu', (e) => this.handleContextMenu(e));
        document.addEventListener('click', () => this.hideContextMenu());
        this.contextMenu.querySelectorAll('.context-menu-item').forEach(item => {
            item.addEventListener('click', (e) => this.handleContextMenuAction(e));
        });

        // Note modal
        this.saveNoteBtn.addEventListener('click', () => this.saveNote());

        // Focus mode
        this.toggleFocusBtn.addEventListener('click', () => this.enterFocusMode());
        this.exitFocusBtn.addEventListener('click', () => this.exitFocusMode());
        this.focusPrevBtn.addEventListener('click', () => this.focusPrevSentence());
        this.focusNextBtn.addEventListener('click', () => this.focusNextSentence());
        this.focusPlayBtn.addEventListener('click', () => this.focusTogglePlay());
        this.focusAutoplayCheckbox.addEventListener('change', (e) => {
            this.focusAutoplay = e.target.checked;
        });

        // Settings
        if (this.toggleSettingsBtn) {
            this.toggleSettingsBtn.addEventListener('click', () => this.toggleSettings());
        }
        if (this.closeSettingsBtn) {
            this.closeSettingsBtn.addEventListener('click', () => this.toggleSettings());
        }
        if (this.bionicToggle) {
            this.bionicToggle.addEventListener('change', (e) => this.toggleBionic(e.target.checked));
        }

        // Theme buttons
        document.querySelectorAll('.theme-btn').forEach(btn => {
            btn.addEventListener('click', () => this.setTheme(btn.dataset.theme));
        });

        // Session timer buttons
        document.querySelectorAll('.session-btn').forEach(btn => {
            btn.addEventListener('click', () => this.startSession(parseInt(btn.dataset.minutes)));
        });

        // Keyboard navigation for focus mode
        document.addEventListener('keydown', (e) => this.handleKeyDown(e));

        // Language/Translation
        if (this.motherTongueSelect) {
            // Load saved language preference
            if (this.motherTongue) {
                this.motherTongueSelect.value = this.motherTongue;
            }
            this.motherTongueSelect.addEventListener('change', (e) => {
                this.motherTongue = e.target.value;
                localStorage.setItem('motherTongue', this.motherTongue);
            });
        }

        // Translation tooltip - hide on scroll or click
        document.addEventListener('scroll', () => this.hideTranslationTooltip(), true);
        document.addEventListener('click', () => this.hideTranslationTooltip());

        // Apply saved settings
        this.applySettings();
    }

    // =========================================================================
    // WebSocket
    // =========================================================================

    initializeWebSocket() {
        this.socket = io();

        this.socket.on('connect', () => {
            console.log('Connected to server');
        });

        this.socket.on('audio_generating', (data) => {
            this.audioStatus[data.sentence_id] = data.status;
            this.updateSentenceVisualState(data.sentence_id, data.status);
        });

        this.socket.on('audio_ready', (data) => {
            const wasReady = this.audioStatus[data.sentence_id] === 'ready';
            this.audioStatus[data.sentence_id] = data.status;

            if (!wasReady) {
                this.audioReadyCount++;
                this.updateAudioProgress();

                // Update library progress periodically (every 10 new audio files)
                if (this.audioReadyCount % 10 === 0) {
                    this.updateLibraryProgress();
                }
            }

            this.updateSentenceVisualState(data.sentence_id, data.status);

            if (this.currentBook && this.isInCurrentChapter(data.sentence_id)) {
                this.preloadSingleAudio(data.sentence_id);
            }

            if (this.pendingPlaySentenceId === data.sentence_id) {
                this.playSentence(data.sentence_id);
                this.pendingPlaySentenceId = null;

                // Update focus mode state if active
                if (this.focusModeActive) {
                    this.focusIsPlaying = true;
                    this.updateFocusPlayButton();
                    this.updateFocusDisplay();
                }
            }
        });

        this.socket.on('audio_failed', (data) => {
            this.audioStatus[data.sentence_id] = data.status;
            this.updateSentenceVisualState(data.sentence_id, data.status);
        });
    }

    // =========================================================================
    // Ollama Integration
    // =========================================================================

    async initializeOllama() {
        try {
            const response = await fetch('/ollama/status');
            const data = await response.json();

            if (data.connected) {
                this.ollamaConnected = true;
                this.ollamaModels = data.models || [];
                this.ollamaStatus.classList.remove('error');
                this.ollamaStatus.classList.add('connected');

                // Show first available model name
                const modelName = this.ollamaModels.length > 0
                    ? this.ollamaModels[0].split(':')[0]
                    : 'Ollama';
                this.ollamaStatusText.textContent = `${modelName} connected`;
            } else {
                throw new Error('Not connected');
            }
        } catch (error) {
            this.ollamaConnected = false;
            this.ollamaModels = [];
            this.ollamaStatus.classList.remove('connected');
            this.ollamaStatus.classList.add('error');
            this.ollamaStatusText.textContent = 'Ollama not available';
        }
    }

    async sendChatMessage() {
        const message = this.chatInput.value.trim();
        if (!message) return;

        // Add user message to chat
        this.addChatMessage(message, 'user');
        this.chatInput.value = '';

        if (!this.ollamaConnected) {
            this.addChatMessage('Ollama is not connected. Please start Ollama locally.', 'assistant');
            return;
        }

        // Disable send button while processing
        this.sendChatBtn.disabled = true;

        // Add loading indicator
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'chat-message assistant loading';
        loadingDiv.innerHTML = '<span class="typing-indicator"><span></span><span></span><span></span></span>';
        this.chatMessages.appendChild(loadingDiv);
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;

        // Only add book context for book-related questions (not casual chat)
        let prompt = message;
        if (this.currentBook && this.pendingChatContext) {
            // Context was explicitly set (e.g., from "Ask AI" action)
            prompt = this.pendingChatContext + message;
            this.pendingChatContext = null;
        }

        // Run Ollama request in background
        this.fetchOllamaResponse(prompt, loadingDiv);
    }

    async fetchOllamaResponse(prompt, loadingDiv) {
        try {
            // Use the first available model, or default to llama3.2
            const model = this.ollamaModels && this.ollamaModels.length > 0
                ? this.ollamaModels[0]
                : 'llama3.2';

            const response = await fetch('/ollama/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model: model,
                    prompt: prompt
                })
            });

            const data = await response.json();

            // Remove loading indicator
            loadingDiv.remove();

            if (data.success) {
                this.addChatMessage(data.response, 'assistant');
            } else {
                this.addChatMessage(data.error || 'Error communicating with Ollama.', 'assistant');
            }
        } catch (error) {
            loadingDiv.remove();
            this.addChatMessage('Error communicating with server.', 'assistant');
        } finally {
            this.sendChatBtn.disabled = false;
        }
    }

    addChatMessage(text, role) {
        const div = document.createElement('div');
        div.className = `chat-message ${role}`;

        if (role === 'assistant') {
            // Render markdown for assistant messages
            div.innerHTML = this.renderMarkdown(text);
        } else {
            div.textContent = text;
        }

        this.chatMessages.appendChild(div);
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    renderMarkdown(text) {
        // Simple markdown renderer
        let html = text
            // Escape HTML first
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            // Bold: **text** or __text__
            .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
            .replace(/__(.+?)__/g, '<strong>$1</strong>')
            // Italic: *text* or _text_
            .replace(/\*([^*]+)\*/g, '<em>$1</em>')
            .replace(/_([^_]+)_/g, '<em>$1</em>')
            // Code: `text`
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            // Line breaks
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>');

        return `<p>${html}</p>`;
    }

    clearChat() {
        this.chatMessages.innerHTML = '';
    }

    // =========================================================================
    // Sidebar Toggles
    // =========================================================================

    showChaptersView() {
        // Show sidebar if hidden
        this.leftSidebar.classList.remove('hidden');

        // Switch to chapters view
        this.chaptersView.style.display = 'flex';
        this.notesView.style.display = 'none';
        this.currentSidebarView = 'chapters';

        // Update button states
        this.toggleChaptersBtn.classList.add('active');
        this.toggleNotesBtn.classList.remove('active');
    }

    showNotesView() {
        // Show sidebar if hidden
        this.leftSidebar.classList.remove('hidden');

        // Switch to notes view
        this.chaptersView.style.display = 'none';
        this.notesView.style.display = 'flex';
        this.currentSidebarView = 'notes';

        // Update button states
        this.toggleChaptersBtn.classList.remove('active');
        this.toggleNotesBtn.classList.add('active');

        // Render notes
        this.renderNotesList();
    }

    toggleLeftSidebar() {
        this.leftSidebar.classList.toggle('hidden');
    }

    toggleChat() {
        if (!this.chatSidebar) return;
        this.chatSidebar.classList.toggle('hidden');
    }

    renderNotesList() {
        const noteKeys = Object.keys(this.notes);
        this.notesCount.textContent = noteKeys.length;

        if (noteKeys.length === 0) {
            this.notesList.innerHTML = `
                <div class="notes-empty">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                        <polyline points="14 2 14 8 20 8"/>
                    </svg>
                    <p>No notes yet</p>
                    <p style="font-size: 0.8rem; margin-top: 0.5rem;">Select text and right-click to add a note</p>
                </div>
            `;
            return;
        }

        this.notesList.innerHTML = '';

        // Sort notes by creation date (newest first)
        const sortedKeys = noteKeys.sort((a, b) => {
            const noteA = this.notes[a];
            const noteB = this.notes[b];
            return new Date(noteB.createdAt || 0) - new Date(noteA.createdAt || 0);
        });

        sortedKeys.forEach(noteKey => {
            const note = this.notes[noteKey];
            // Handle both old format (string) and new format (object)
            const selectedText = typeof note === 'object' ? note.selectedText : null;
            const noteContent = typeof note === 'object' ? note.content : note;
            const sentenceId = typeof note === 'object' ? note.sentenceId : noteKey;

            const displayText = selectedText || 'Selected text';

            const noteItem = document.createElement('div');
            noteItem.className = 'note-item';
            noteItem.innerHTML = `
                <div class="note-item-text">"${this.truncateText(displayText, 80)}"</div>
                <div class="note-item-content">${this.truncateText(noteContent, 100)}</div>
                <div class="note-item-actions">
                    <button class="icon-btn-small" title="Delete note" data-action="delete">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                        </svg>
                    </button>
                </div>
            `;

            // Click to navigate to sentence (not updating read position)
            noteItem.addEventListener('click', (e) => {
                if (e.target.closest('[data-action="delete"]')) {
                    e.stopPropagation();
                    this.deleteNote(noteKey);
                    return;
                }
                this.navigateToSentence(sentenceId, false); // false = don't update bookmark
            });

            this.notesList.appendChild(noteItem);
        });
    }

    truncateText(text, maxLength) {
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength) + '...';
    }

    navigateToSentence(sentenceId, updateBookmark = true) {
        // Find which chapter contains this sentence
        const sentence = this.sentences.find(s => s.id === sentenceId);
        if (!sentence) return;

        // Find chapter index
        let targetChapterIndex = -1;
        for (let i = 0; i < this.currentBook.chapters.length; i++) {
            const chapter = this.currentBook.chapters[i];
            if (chapter.sentences && chapter.sentences.some(s => s.id === sentenceId)) {
                targetChapterIndex = i;
                break;
            }
        }

        // If not found in chapter data, check by searching HTML
        if (targetChapterIndex === -1) {
            for (let i = 0; i < this.currentBook.chapters.length; i++) {
                const chapter = this.currentBook.chapters[i];
                if (chapter.html && chapter.html.includes(`data-sentence-id="${sentenceId}"`)) {
                    targetChapterIndex = i;
                    break;
                }
            }
        }

        // Navigate to chapter if different (don't save position if from notes)
        if (targetChapterIndex !== -1 && targetChapterIndex !== this.currentChapter) {
            this.displayChapter(targetChapterIndex, updateBookmark);
        }

        // Scroll to and highlight the sentence
        setTimeout(() => {
            const el = document.querySelector(`[data-sentence-id="${sentenceId}"]`);
            if (el) {
                el.scrollIntoView({ behavior: 'smooth', block: 'center' });
                el.classList.add('highlight-pulse');
                setTimeout(() => el.classList.remove('highlight-pulse'), 2000);
            }
        }, 300);
    }

    deleteNote(noteKey) {
        if (confirm('Delete this note?')) {
            const note = this.notes[noteKey];
            const sentenceId = typeof note === 'object' ? note.sentenceId : noteKey;

            delete this.notes[noteKey];

            // Only remove visual if no other notes on this sentence
            const hasOtherNotes = Object.values(this.notes).some(n =>
                (typeof n === 'object' ? n.sentenceId : null) === sentenceId
            );
            if (!hasOtherNotes) {
                const el = document.querySelector(`[data-sentence-id="${sentenceId}"]`);
                if (el) el.classList.remove('has-note');
            }

            // Save and re-render
            if (this.currentBook) {
                localStorage.setItem(`notes_${this.currentBook.book_id}`, JSON.stringify(this.notes));
            }
            this.renderNotesList();
        }
    }

    exportNotes() {
        if (Object.keys(this.notes).length === 0) {
            alert('No notes to export');
            return;
        }

        let content = `Notes from "${this.currentBook?.title || 'Book'}"\n`;
        content += `Exported on ${new Date().toLocaleDateString()}\n`;
        content += '='.repeat(50) + '\n\n';

        Object.values(this.notes).forEach(note => {
            const selectedText = typeof note === 'object' ? note.selectedText : 'Unknown';
            const noteContent = typeof note === 'object' ? note.content : note;

            content += `"${selectedText}"\n`;
            content += `Note: ${noteContent}\n`;
            content += '-'.repeat(30) + '\n\n';
        });

        // Download as text file
        const blob = new Blob([content], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `notes_${this.currentBook?.title?.replace(/\s+/g, '_') || 'book'}.txt`;
        a.click();
        URL.revokeObjectURL(url);
    }

    // =========================================================================
    // Context Menu
    // =========================================================================

    handleContextMenu(e) {
        const sentence = e.target.closest('.sentence');
        if (!sentence) return;

        e.preventDefault();
        this.selectedSentenceId = sentence.dataset.sentenceId;

        // Capture selected text if any
        const selection = window.getSelection();
        this.selectedText = selection && selection.toString().trim()
            ? selection.toString().trim()
            : null;

        this.contextMenu.style.display = 'block';
        this.contextMenu.style.left = `${e.pageX}px`;
        this.contextMenu.style.top = `${e.pageY}px`;
    }

    hideContextMenu() {
        this.contextMenu.style.display = 'none';
    }

    handleContextMenuAction(e) {
        const action = e.currentTarget.dataset.action;

        switch (action) {
            case 'copy':
                this.copySelectedText();
                break;
            case 'addNote':
                this.openNoteModal();
                break;
            case 'askAI':
                this.askAIAboutSentence();
                break;
            case 'playThis':
                if (this.selectedSentenceId) {
                    this.handleSentenceClick(this.selectedSentenceId);
                }
                break;
        }

        this.hideContextMenu();
    }

    copySelectedText() {
        // Get selected text or sentence text
        const textToCopy = this.selectedText ||
            (this.selectedSentenceId ?
                this.sentences.find(s => s.id === this.selectedSentenceId)?.text : '');

        if (textToCopy) {
            navigator.clipboard.writeText(textToCopy).then(() => {
                // Brief visual feedback
                this.showToast('Copied to clipboard');
            }).catch(err => {
                console.error('Failed to copy:', err);
            });
        }
    }

    showToast(message) {
        // Create a simple toast notification
        const toast = document.createElement('div');
        toast.className = 'toast-notification';
        toast.textContent = message;
        toast.style.cssText = `
            position: fixed;
            bottom: 100px;
            left: 50%;
            transform: translateX(-50%);
            background: var(--text);
            color: var(--bg-content);
            padding: 0.75rem 1.5rem;
            border-radius: var(--radius);
            font-size: 0.9rem;
            z-index: 10000;
            animation: fadeInOut 2s ease;
        `;
        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 2000);
    }

    // =========================================================================
    // Translation
    // =========================================================================

    setupWordHoverTranslation() {
        // Only setup if a language is selected
        if (!this.motherTongue || !this.chapterContent) return;

        // Add mouseover listener to chapter content
        this.chapterContent.addEventListener('mouseover', (e) => this.handleWordHover(e));
        this.chapterContent.addEventListener('mouseout', (e) => this.handleWordHoverOut(e));
    }

    handleWordHover(e) {
        // Only translate if language is set
        if (!this.motherTongue) return;

        // Clear any pending hover timeout
        if (this.hoverTimeout) {
            clearTimeout(this.hoverTimeout);
        }

        // Get the word under cursor
        const word = this.getWordAtPoint(e.clientX, e.clientY);
        if (!word || word.length < 2) return;

        // Delay translation to avoid too many requests
        this.hoverTimeout = setTimeout(() => {
            this.translateAndShowTooltip(word, e.clientX, e.clientY);
        }, 500); // 500ms delay
    }

    handleWordHoverOut(e) {
        if (this.hoverTimeout) {
            clearTimeout(this.hoverTimeout);
            this.hoverTimeout = null;
        }
    }

    getWordAtPoint(x, y) {
        // Use document.caretPositionFromPoint or caretRangeFromPoint
        let range;
        if (document.caretPositionFromPoint) {
            const pos = document.caretPositionFromPoint(x, y);
            if (!pos) return null;
            range = document.createRange();
            range.setStart(pos.offsetNode, pos.offset);
            range.setEnd(pos.offsetNode, pos.offset);
        } else if (document.caretRangeFromPoint) {
            range = document.caretRangeFromPoint(x, y);
        }

        if (!range) return null;

        // Expand range to word boundaries
        const node = range.startContainer;
        if (node.nodeType !== Node.TEXT_NODE) return null;

        const text = node.textContent;
        const offset = range.startOffset;

        // Find word boundaries
        let start = offset, end = offset;
        while (start > 0 && /\w/.test(text[start - 1])) start--;
        while (end < text.length && /\w/.test(text[end])) end++;

        const word = text.slice(start, end).trim();

        // Only return alphabetic words (ignore numbers, punctuation)
        if (/^[a-zA-Z]+$/.test(word)) {
            return word;
        }
        return null;
    }

    async translateAndShowTooltip(word, x, y) {
        if (!word || !this.motherTongue) return;

        const cacheKey = `${word.toLowerCase()}_${this.motherTongue}`;

        // Check cache first
        if (this.translationCache[cacheKey]) {
            this.showTranslationTooltip(word, this.translationCache[cacheKey], x, y);
            return;
        }

        // Show loading state
        this.showTranslationTooltipLoading(word, x, y);

        try {
            const response = await fetch('/translate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: word,
                    target: this.motherTongue,
                    source: 'en'
                })
            });

            const data = await response.json();

            if (data.translation && data.translation.toLowerCase() !== word.toLowerCase()) {
                // Cache the result
                this.translationCache[cacheKey] = data.translation;
                this.showTranslationTooltip(word, data.translation, x, y);
            } else {
                // Same word or error - hide tooltip
                this.hideTranslationTooltip();
            }
        } catch (error) {
            console.error('Translation error:', error);
            this.hideTranslationTooltip();
        }
    }

    showTranslationTooltipLoading(word, x, y) {
        if (!this.translationTooltip) return;

        const tooltip = this.translationTooltip;
        tooltip.querySelector('.tooltip-original').textContent = word;
        tooltip.querySelector('.tooltip-translation').textContent = '';
        tooltip.querySelector('.tooltip-arrow').style.display = 'none';
        tooltip.querySelector('.tooltip-loading').style.display = 'flex';

        this.positionTooltip(tooltip, x, y);
        tooltip.style.display = 'flex';
    }

    showTranslationTooltip(original, translation, x, y) {
        if (!this.translationTooltip) return;

        const tooltip = this.translationTooltip;
        tooltip.querySelector('.tooltip-original').textContent = original;
        tooltip.querySelector('.tooltip-translation').textContent = translation;
        tooltip.querySelector('.tooltip-arrow').style.display = 'block';
        tooltip.querySelector('.tooltip-loading').style.display = 'none';

        this.positionTooltip(tooltip, x, y);
        tooltip.style.display = 'flex';

        // Auto-hide after 3 seconds
        setTimeout(() => this.hideTranslationTooltip(), 3000);
    }

    positionTooltip(tooltip, x, y) {
        // Position above the cursor
        const tooltipRect = tooltip.getBoundingClientRect();
        let left = x - 50;
        let top = y - 50;

        // Keep within viewport
        if (left < 10) left = 10;
        if (left + 200 > window.innerWidth) left = window.innerWidth - 210;
        if (top < 10) top = y + 20; // Show below if too close to top

        tooltip.style.left = `${left}px`;
        tooltip.style.top = `${top}px`;
    }

    hideTranslationTooltip() {
        if (this.translationTooltip) {
            this.translationTooltip.style.display = 'none';
        }
    }

    // =========================================================================
    // Notes
    // =========================================================================

    openNoteModal() {
        if (!this.selectedSentenceId) return;

        const sentence = this.sentences.find(s => s.id === this.selectedSentenceId);
        if (!sentence) return;

        // Use selected text if available, otherwise use full sentence
        const textForNote = this.selectedText || sentence.text;
        this.noteSelectedText.textContent = textForNote;

        // Store the text we're making a note about
        this.pendingNoteText = textForNote;

        // Check if there's an existing note for this exact text
        const noteKey = this.getNoteKey(this.selectedSentenceId, textForNote);
        const existingNote = this.notes[noteKey];
        this.noteInput.value = existingNote ? existingNote.content : '';

        this.noteModal.style.display = 'flex';
    }

    getNoteKey(sentenceId, text) {
        // Create a unique key for each note based on sentence and selected text
        return `${sentenceId}:${text.substring(0, 50)}`;
    }

    saveNote() {
        if (!this.selectedSentenceId || !this.pendingNoteText) return;

        const noteContent = this.noteInput.value.trim();
        const noteKey = this.getNoteKey(this.selectedSentenceId, this.pendingNoteText);

        if (noteContent) {
            this.notes[noteKey] = {
                sentenceId: this.selectedSentenceId,
                selectedText: this.pendingNoteText,
                content: noteContent,
                createdAt: new Date().toISOString()
            };
            const el = document.querySelector(`[data-sentence-id="${this.selectedSentenceId}"]`);
            if (el) el.classList.add('has-note');
        } else {
            delete this.notes[noteKey];
            // Only remove visual if no other notes on this sentence
            const hasOtherNotes = Object.values(this.notes).some(n => n.sentenceId === this.selectedSentenceId);
            if (!hasOtherNotes) {
                const el = document.querySelector(`[data-sentence-id="${this.selectedSentenceId}"]`);
                if (el) el.classList.remove('has-note');
            }
        }

        // Save notes to localStorage
        if (this.currentBook) {
            localStorage.setItem(`notes_${this.currentBook.book_id}`, JSON.stringify(this.notes));
        }

        // Update notes count badge
        this.notesCount.textContent = Object.keys(this.notes).length;

        // Re-render notes list if visible
        if (this.currentSidebarView === 'notes') {
            this.renderNotesList();
        }

        this.pendingNoteText = null;
        this.noteModal.style.display = 'none';
    }

    loadNotes() {
        if (!this.currentBook) return;
        const saved = localStorage.getItem(`notes_${this.currentBook.book_id}`);
        if (saved) {
            const parsed = JSON.parse(saved);
            // Migrate old format notes (string) to new format (object)
            this.notes = {};
            Object.entries(parsed).forEach(([key, value]) => {
                if (typeof value === 'string') {
                    // Old format: key was sentenceId, value was note text
                    const sentence = this.sentences.find(s => s.id === key);
                    this.notes[key] = {
                        sentenceId: key,
                        selectedText: sentence ? sentence.text : 'Unknown',
                        content: value,
                        createdAt: new Date().toISOString()
                    };
                } else {
                    // New format
                    this.notes[key] = value;
                }
            });
        } else {
            this.notes = {};
        }
        this.notesCount.textContent = Object.keys(this.notes).length;
    }

    askAIAboutSentence() {
        if (!this.selectedSentenceId) return;

        // Use selected text if available, otherwise use full sentence
        const selectedText = this.selectedText || null;
        const sentence = this.sentences.find(s => s.id === this.selectedSentenceId);

        const textToExplain = selectedText || (sentence ? sentence.text : null);
        if (!textToExplain) return;

        // Open chat sidebar
        this.chatSidebar.classList.remove('hidden');

        // Set context for this specific question
        const bookContext = this.currentBook ? `From the book "${this.currentBook.title}": ` : '';
        this.pendingChatContext = `${bookContext}The user wants to understand this passage: "${textToExplain}"\n\nUser's question: `;

        // Pre-fill with a question
        this.chatInput.value = `Explain this passage`;
        this.chatInput.focus();
    }

    // =========================================================================
    // File Upload & Book Loading
    // =========================================================================

    async handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        this.showLoading('Opening book...');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                await this.loadBookContent();
            } else {
                alert('Error: ' + data.error);
            }
        } catch (error) {
            console.error('Upload error:', error);
            alert('Failed to upload file');
        } finally {
            this.hideLoading();
        }
    }

    async loadBookContent() {
        const response = await fetch('/book/content');
        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        this.currentBook = data;
        this.sentences = data.sentences || [];
        this.audioStatus = data.audio_status || {};
        this.totalSentences = this.sentences.length;

        this.audioReadyCount = Object.values(this.audioStatus)
            .filter(status => status === 'ready').length;

        this.loadNotes();
        this.displayBook();
        this.prioritizeCurrentChapter();
        this.startAudioGeneration();
    }

    displayBook() {
        this.welcomeScreen.style.display = 'none';
        this.readerContainer.style.display = 'flex';
        this.audioControls.style.display = 'flex';
        this.clearBtn.style.display = 'inline-block';

        this.headerTitle.textContent = this.currentBook.title;
        this.bookTitle.textContent = this.currentBook.title;

        // Ensure chapters view is shown
        this.showChaptersView();

        this.displayChapterList();

        // Restore position or start at chapter 0
        this.restorePosition();

        this.updateAudioProgress();

        // Setup word hover translation
        this.setupWordHoverTranslation();
    }

    displayChapterList() {
        this.chapterList.innerHTML = '';

        this.currentBook.chapters.forEach((chapter, index) => {
            const li = document.createElement('li');
            li.textContent = chapter.title || `Chapter ${index + 1}`;
            li.dataset.chapterIndex = index;

            if (index === this.currentChapter) li.classList.add('active');

            li.addEventListener('click', () => {
                this.displayChapter(index);
                this.prioritizeCurrentChapter();
            });

            this.chapterList.appendChild(li);
        });
    }

    displayChapter(index, saveBookmark = true, keepPlaying = false) {
        this.currentChapter = index;
        const chapter = this.currentBook.chapters[index];

        if (!chapter) return;

        // Stop playback unless we're transitioning chapters during autoplay
        if (this.currentAudio && this.isPlaying && !keepPlaying) {
            this.stopPlayback();
        }

        // Update active chapter
        document.querySelectorAll('.chapter-list li').forEach(li => {
            li.classList.remove('active');
        });
        const activeEl = document.querySelector(`[data-chapter-index="${index}"]`);
        if (activeEl) activeEl.classList.add('active');

        this.chapterContent.innerHTML = chapter.html || '';

        this.attachSentenceHandlers();
        this.applyNotesHighlights();
        this.updateServerPosition();
        this.preloadVisibleAudio();

        // Only save position if this is actual reading (not note navigation)
        if (saveBookmark) {
            this.savePosition();
        }
    }

    attachSentenceHandlers() {
        const sentenceSpans = this.chapterContent.querySelectorAll('.sentence');

        sentenceSpans.forEach(span => {
            const sentenceId = span.dataset.sentenceId;
            const status = this.audioStatus[sentenceId];

            if (status === 'ready') span.classList.add('audio-ready');
            else if (status === 'pending') span.classList.add('audio-pending');
            else if (status === 'generating') span.classList.add('audio-generating');

            span.addEventListener('click', (e) => {
                e.stopPropagation();
                this.handleSentenceClick(sentenceId);
            });
        });
    }

    applyNotesHighlights() {
        // Get all unique sentence IDs that have notes
        const sentenceIdsWithNotes = new Set();
        Object.values(this.notes).forEach(note => {
            const sentenceId = typeof note === 'object' ? note.sentenceId : null;
            if (sentenceId) sentenceIdsWithNotes.add(sentenceId);
        });

        sentenceIdsWithNotes.forEach(sentenceId => {
            const el = document.querySelector(`[data-sentence-id="${sentenceId}"]`);
            if (el) el.classList.add('has-note');
        });
    }

    updateSentenceVisualState(sentenceId, status) {
        const el = document.querySelector(`[data-sentence-id="${sentenceId}"]`);
        if (el) {
            el.classList.remove('audio-pending', 'audio-generating', 'audio-ready', 'audio-failed');
            if (status) el.classList.add(`audio-${status}`);
        }
    }

    // =========================================================================
    // Audio Playback
    // =========================================================================

    async handleSentenceClick(sentenceId) {
        const status = this.audioStatus[sentenceId];

        if (status === 'ready') {
            await this.playSentence(sentenceId);
        } else if (status === 'pending') {
            this.pendingPlaySentenceId = sentenceId;
            await this.prioritizeSentences([sentenceId]);
        } else if (status === 'generating') {
            this.pendingPlaySentenceId = sentenceId;
        }
    }

    async playSentence(sentenceId) {
        const status = this.audioStatus[sentenceId];
        if (status !== 'ready') return;

        // Skip if already playing this sentence
        if (this.currentSentence && this.currentSentence.id === sentenceId && (this.isPlaying || this.focusIsPlaying)) {
            return;
        }

        // CRITICAL: Stop all audio before playing new sentence
        this.stopAllAudio();

        const sentence = this.sentences.find(s => s.id === sentenceId);
        if (!sentence) return;

        this.currentSentence = sentence;

        // Highlight ALL spans with this sentence ID (sentence may span multiple spans)
        const sentenceElements = document.querySelectorAll(`[data-sentence-id="${sentenceId}"]`);
        sentenceElements.forEach(el => el.classList.add('playing'));

        // Scroll to the first element
        if (sentenceElements.length > 0) {
            sentenceElements[0].scrollIntoView({ behavior: 'smooth', block: 'center' });
        }

        if (!this.audioElements[sentenceId]) {
            await this.preloadSingleAudio(sentenceId);
        }

        const audio = this.audioElements[sentenceId];
        if (audio) {
            this.currentAudio = audio;
            audio.playbackRate = this.playbackRate;
            audio.currentTime = 0;

            try {
                await audio.play();
                this.isPlaying = true;
                this.updatePlayPauseButton();

                // Update focus mode if active
                if (this.focusModeActive) {
                    const idx = this.focusSentences.findIndex(s => s.id === sentenceId);
                    if (idx >= 0) {
                        this.focusCurrentIndex = idx;
                        this.updateFocusDisplay();
                    }
                }
            } catch (error) {
                console.error('Error playing audio:', error);
            }
        }
    }

    async preloadSingleAudio(sentenceId) {
        if (this.audioElements[sentenceId]) return;

        const bookId = this.currentBook.book_id;
        const audioUrl = `/audio/${bookId}/${sentenceId}`;

        const audio = new Audio(audioUrl);
        audio.preload = 'auto';
        audio.playbackRate = this.playbackRate;

        audio.addEventListener('ended', () => this.onAudioEnded());
        audio.addEventListener('error', (e) => console.error(`Audio error for ${sentenceId}:`, e));

        this.audioElements[sentenceId] = audio;
    }

    async preloadVisibleAudio() {
        const sentences = this.chapterContent.querySelectorAll('.sentence');
        const visibleSentences = Array.from(sentences).slice(0, 20);

        for (const element of visibleSentences) {
            const sentenceId = element.dataset.sentenceId;
            if (this.audioStatus[sentenceId] === 'ready') {
                await this.preloadSingleAudio(sentenceId);
            }
        }
    }

    onAudioEnded() {
        if (this.isTransitioning) return;
        this.isTransitioning = true;

        document.querySelectorAll('.sentence.playing').forEach(el => {
            el.classList.remove('playing');
            el.classList.add('completed');
        });

        // Handle focus mode audio separately
        if (this.focusModeActive) {
            this.isTransitioning = false;
            this.onFocusAudioEnded();
            return;
        }

        // Normal mode autoplay
        if (this.autoPlay && this.currentSentence) {
            setTimeout(() => {
                this.isTransitioning = false;
                this.playNextSentence();
            }, 100);
        } else {
            this.isPlaying = false;
            this.isTransitioning = false;
            this.updatePlayPauseButton();
        }
    }

    async playNextSentence() {
        const currentChapterSentences = this.getCurrentChapterSentences();

        if (currentChapterSentences.length === 0) {
            this.stopPlayback();
            return;
        }

        if (!this.currentSentence) {
            const firstReady = currentChapterSentences.find(s => this.audioStatus[s.id] === 'ready');
            if (firstReady) await this.playSentence(firstReady.id);
            return;
        }

        const currentId = this.currentSentence.id;
        const currentIndex = currentChapterSentences.findIndex(s => s.id === currentId);

        if (currentIndex === -1) {
            const firstReady = currentChapterSentences.find(s => this.audioStatus[s.id] === 'ready');
            if (firstReady) await this.playSentence(firstReady.id);
            return;
        }

        // Look for next ready sentence in current chapter
        for (let i = currentIndex + 1; i < currentChapterSentences.length; i++) {
            const nextSentence = currentChapterSentences[i];
            if (nextSentence && this.audioStatus[nextSentence.id] === 'ready') {
                await this.playSentence(nextSentence.id);
                return;
            }
            // If next sentence is generating/pending, wait for it
            if (nextSentence && (this.audioStatus[nextSentence.id] === 'generating' || this.audioStatus[nextSentence.id] === 'pending')) {
                this.pendingPlaySentenceId = nextSentence.id;
                // Keep playing state so UI shows we're waiting
                this.isPlaying = true;
                // Prioritize this sentence
                this.prioritizeSentences([nextSentence.id]);
                return;
            }
        }

        // End of chapter - try to move to next chapter if autoplay is on
        if (this.autoPlay && this.currentChapter < this.currentBook.chapters.length - 1) {
            // Move to next chapter (keepPlaying=true to not stop playback)
            this.displayChapter(this.currentChapter + 1, true, true);
            // Wait a bit for chapter to load, then start playing
            setTimeout(() => {
                const nextChapterSentences = this.getCurrentChapterSentences();
                const firstReady = nextChapterSentences.find(s => this.audioStatus[s.id] === 'ready');
                if (firstReady) {
                    this.playSentence(firstReady.id);
                } else if (nextChapterSentences.length > 0) {
                    // Wait for first sentence to be ready
                    this.pendingPlaySentenceId = nextChapterSentences[0].id;
                    this.isPlaying = true; // Keep playing state
                    this.prioritizeSentences([nextChapterSentences[0].id]);
                }
            }, 300);
            return;
        }

        // No more sentences to play
        this.isPlaying = false;
        this.currentSentence = null;
        this.updatePlayPauseButton();
    }

    async playPreviousSentence() {
        if (!this.currentSentence) return;

        const currentChapterSentences = this.getCurrentChapterSentences();
        const currentIndex = currentChapterSentences.findIndex(s => s.id === this.currentSentence.id);

        for (let i = currentIndex - 1; i >= 0; i--) {
            if (this.audioStatus[currentChapterSentences[i].id] === 'ready') {
                await this.playSentence(currentChapterSentences[i].id);
                return;
            }
        }
    }

    togglePlayPause() {
        if (this.currentAudio && this.isPlaying) {
            this.currentAudio.pause();
            this.isPlaying = false;
        } else if (this.currentAudio) {
            this.currentAudio.play();
            this.isPlaying = true;
            if (this.currentSentence) {
                // Highlight ALL spans with this sentence ID
                document.querySelectorAll(`[data-sentence-id="${this.currentSentence.id}"]`)
                    .forEach(el => el.classList.add('playing'));
            }
        } else {
            this.playNextSentence();
        }
        this.updatePlayPauseButton();
    }

    // Stop ALL audio - ensures only one audio plays at a time
    stopAllAudio() {
        // Stop current audio
        if (this.currentAudio) {
            this.currentAudio.pause();
            this.currentAudio.currentTime = 0;
        }

        // Stop all preloaded audio elements
        Object.values(this.audioElements).forEach(audio => {
            audio.pause();
            audio.currentTime = 0;
        });

        // Reset all playing states
        this.isPlaying = false;
        this.focusIsPlaying = false;
        this.isTransitioning = false;

        document.querySelectorAll('.sentence.playing').forEach(el => {
            el.classList.remove('playing');
        });
    }

    stopPlayback() {
        this.stopAllAudio();
        this.currentSentence = null;

        document.querySelectorAll('.sentence.completed').forEach(el => {
            el.classList.remove('completed');
        });

        this.updatePlayPauseButton();
    }

    updatePlaybackSpeed(event) {
        this.playbackRate = parseFloat(event.target.value);
        this.speedValue.textContent = `${this.playbackRate}x`;

        Object.values(this.audioElements).forEach(audio => {
            audio.playbackRate = this.playbackRate;
        });
    }

    toggleAutoplay(event) {
        this.autoPlay = event.target.checked;
        if (this.autoPlay && !this.isPlaying) {
            this.playNextSentence();
        }
    }

    updatePlayPauseButton() {
        const playIcon = this.playPauseBtn.querySelector('.icon-play');
        const pauseIcon = this.playPauseBtn.querySelector('.icon-pause');

        if (this.isPlaying) {
            playIcon.style.display = 'none';
            pauseIcon.style.display = 'inline';
        } else {
            playIcon.style.display = 'inline';
            pauseIcon.style.display = 'none';
        }
    }

    // =========================================================================
    // Position Tracking
    // =========================================================================

    startPositionTracking() {
        setInterval(() => this.savePosition(), 10000);

        document.addEventListener('visibilitychange', () => {
            if (document.hidden) this.savePosition();
        });

        window.addEventListener('beforeunload', () => this.savePositionSync());
    }

    getCurrentPosition() {
        return {
            chapter_index: this.currentChapter,
            sentence_id: this.currentSentence?.id || null,
            sentence_index: this.currentSentence?.sentence_index || 0
        };
    }

    async savePosition() {
        if (!this.currentBook) return;

        const position = this.getCurrentPosition();

        try {
            await fetch('/position/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    book_id: this.currentBook.book_id,
                    position: position
                })
            });

            localStorage.setItem(`epub_position_${this.currentBook.book_id}`, JSON.stringify(position));

            // Also update library progress
            this.updateLibraryProgress();
        } catch (error) {
            console.error('Error saving position:', error);
        }
    }

    savePositionSync() {
        if (!this.currentBook) return;

        const position = this.getCurrentPosition();
        localStorage.setItem(`epub_position_${this.currentBook.book_id}`, JSON.stringify(position));

        navigator.sendBeacon('/position/save', new Blob([JSON.stringify({
            book_id: this.currentBook.book_id,
            position: position
        })], { type: 'application/json' }));
    }

    async loadSavedPosition() {
        if (!this.currentBook) return null;

        try {
            const response = await fetch(`/position/load/${this.currentBook.book_id}`);
            const data = await response.json();
            if (data.success && data.position) return data.position;
        } catch (error) {
            console.error('Error loading position:', error);
        }

        const local = localStorage.getItem(`epub_position_${this.currentBook.book_id}`);
        return local ? JSON.parse(local) : null;
    }

    async restorePosition() {
        const position = await this.loadSavedPosition();

        if (position && position.chapter_index !== undefined) {
            this.displayChapter(position.chapter_index);

            if (position.sentence_id) {
                setTimeout(() => {
                    const el = document.querySelector(`[data-sentence-id="${position.sentence_id}"]`);
                    if (el) {
                        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
                        el.classList.add('playing');
                        setTimeout(() => el.classList.remove('playing'), 2000);
                    }
                }, 500);
            }
        } else if (this.currentBook.chapters.length > 0) {
            this.displayChapter(0);
        }
    }

    // =========================================================================
    // Utilities
    // =========================================================================

    getCurrentChapterSentences() {
        if (!this.sentences || this.sentences.length === 0) return [];

        const sentenceElements = this.chapterContent.querySelectorAll('.sentence');
        // Get UNIQUE sentence IDs (a sentence can span multiple spans)
        const chapterSentenceIds = [...new Set(
            Array.from(sentenceElements).map(el => el.dataset.sentenceId)
        )];

        // Return sentences in the order they appear in the chapter
        return chapterSentenceIds
            .map(id => this.sentences.find(s => s.id === id))
            .filter(s => s !== undefined);
    }

    isInCurrentChapter(sentenceId) {
        if (!this.currentBook || !this.currentBook.chapters) return false;
        const chapter = this.currentBook.chapters[this.currentChapter];
        return chapter && chapter.sentences && chapter.sentences.some(s => s.id === sentenceId);
    }

    updateAudioProgress() {
        this.audioReady.textContent = this.audioReadyCount;
        this.audioTotal.textContent = this.totalSentences;

        const progress = this.totalSentences > 0
            ? (this.audioReadyCount / this.totalSentences * 100)
            : 0;

        this.audioProgressBar.style.width = `${Math.min(100, progress)}%`;
    }

    async prioritizeCurrentChapter() {
        this.socket.emit('request_chapter_audio', { chapter_index: this.currentChapter });
    }

    async updateServerPosition() {
        if (!this.currentBook) return;

        const visibleSentences = this.chapterContent.querySelectorAll('.sentence');
        const pageSentenceIds = Array.from(visibleSentences).slice(0, 30).map(el => el.dataset.sentenceId);

        try {
            await fetch('/update_position', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    chapter_index: this.currentChapter,
                    page_sentences: pageSentenceIds
                })
            });

            if (pageSentenceIds.length > 0) {
                await this.prioritizeSentences(pageSentenceIds);
            }
        } catch (error) {
            console.error('Error updating position:', error);
        }
    }

    async prioritizeSentences(sentenceIds) {
        try {
            await fetch('/prioritize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sentence_ids: sentenceIds })
            });
        } catch (error) {
            console.error('Error prioritizing sentences:', error);
        }
    }

    async startAudioGeneration() {
        if (!this.currentBook) return;

        try {
            await fetch('/start_generation', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ book_id: this.currentBook.book_id })
            });
        } catch (error) {
            console.error('Error starting audio generation:', error);
        }
    }

    async clearBook() {
        if (confirm('Close this book?')) {
            // Update library progress before closing
            await this.updateLibraryProgress();

            this.stopPlayback();

            this.currentBook = null;
            this.sentences = [];
            this.audioElements = {};
            this.audioStatus = {};
            this.audioReadyCount = 0;
            this.totalSentences = 0;
            this.notes = {};

            this.welcomeScreen.style.display = 'flex';
            this.readerContainer.style.display = 'none';
            this.audioControls.style.display = 'none';
            this.clearBtn.style.display = 'none';
            this.fileInput.value = '';
            this.headerTitle.textContent = 'EPUB Reader';

            // Reset sidebar to chapters view
            this.showChaptersView();
            this.notesCount.textContent = '0';
            this.toggleChaptersBtn.classList.remove('active');
            this.toggleNotesBtn.classList.remove('active');

            await fetch('/clear');

            // Refresh library view
            await this.loadLibrary();
        }
    }

    showLoading(text) {
        this.loadingOverlay.style.display = 'flex';
        this.loadingText.textContent = text;
    }

    hideLoading() {
        this.loadingOverlay.style.display = 'none';
    }

    // =========================================================================
    // Focus Mode
    // =========================================================================

    enterFocusMode() {
        if (!this.currentBook) return;

        // Get sentences from current chapter
        this.focusSentences = this.getCurrentChapterSentences();

        if (this.focusSentences.length === 0) return;

        // Save current position to return to when exiting
        if (this.currentSentence) {
            this.focusReturnSentenceId = this.currentSentence.id;
        }

        // CRITICAL: Stop ALL audio before entering focus mode
        this.stopAllAudio();
        this.updatePlayPauseButton();

        // Start from current sentence if one was playing/selected, otherwise from beginning
        if (this.currentSentence) {
            const idx = this.focusSentences.findIndex(s => s.id === this.currentSentence.id);
            this.focusCurrentIndex = idx >= 0 ? idx : 0;
        } else {
            this.focusCurrentIndex = 0;
        }

        // Reset focus audio state
        this.focusIsPlaying = false;

        this.focusModeActive = true;
        this.focusMode.style.display = 'flex';
        this.updateFocusDisplay();
        this.updateFocusPlayButton();

        // Hide other UI
        document.body.style.overflow = 'hidden';
    }

    exitFocusMode() {
        // CRITICAL: Stop ALL audio when exiting focus mode
        this.stopAllAudio();

        this.focusModeActive = false;
        this.focusMode.style.display = 'none';
        document.body.style.overflow = '';

        // Sync currentSentence with where we were in focus mode
        // This ensures re-entering focus mode continues from the same position
        const focusSentence = this.focusSentences[this.focusCurrentIndex];
        if (focusSentence) {
            this.currentSentence = focusSentence;
        }

        // Navigate to the sentence we were on when exiting
        const returnId = this.currentSentence?.id || this.focusReturnSentenceId;
        if (returnId) {
            const el = document.querySelector(`[data-sentence-id="${returnId}"]`);
            if (el) {
                el.scrollIntoView({ behavior: 'smooth', block: 'center' });
                // Briefly highlight the sentence
                el.classList.add('focus-return-highlight');
                setTimeout(() => el.classList.remove('focus-return-highlight'), 2000);
            }
        }

        this.focusReturnSentenceId = null;
    }

    updateFocusDisplay() {
        const sentence = this.focusSentences[this.focusCurrentIndex];
        if (!sentence) return;

        const text = sentence.text;

        // Apply bionic reading if enabled
        this.focusParagraph.innerHTML = this.bionicEnabled
            ? this.applyBionicReading(text)
            : text;

        // Highlight if currently playing in focus mode
        if (this.focusIsPlaying && this.currentSentence && this.currentSentence.id === sentence.id) {
            this.focusParagraph.classList.add('focus-playing');
        } else {
            this.focusParagraph.classList.remove('focus-playing');
        }

        // Update progress
        const progress = ((this.focusCurrentIndex + 1) / this.focusSentences.length) * 100;
        this.focusProgressBar.style.width = `${progress}%`;
        this.focusProgressText.textContent = `${this.focusCurrentIndex + 1} / ${this.focusSentences.length}`;

        // Update buttons
        this.focusPrevBtn.disabled = this.focusCurrentIndex === 0;
        this.focusNextBtn.textContent = this.focusCurrentIndex === this.focusSentences.length - 1
            ? 'Finish ✓'
            : 'Next →';
    }

    focusNextSentence() {
        if (this.focusCurrentIndex < this.focusSentences.length - 1) {
            this.focusCurrentIndex++;
            // Update return position to track where we are
            this.focusReturnSentenceId = this.focusSentences[this.focusCurrentIndex].id;
            this.updateFocusDisplay();
            this.focusParagraph.style.animation = 'none';
            setTimeout(() => this.focusParagraph.style.animation = 'fadeIn 0.3s ease', 10);
        } else {
            // Completed chapter in focus mode
            this.exitFocusMode();
            this.showCelebration('Chapter Complete!', 'Great focus session!');
            this.updateStats('chapter');
        }
    }

    focusPrevSentence() {
        if (this.focusCurrentIndex > 0) {
            this.focusCurrentIndex--;
            // Update return position to track where we are
            this.focusReturnSentenceId = this.focusSentences[this.focusCurrentIndex].id;
            this.updateFocusDisplay();
            this.focusParagraph.style.animation = 'none';
            setTimeout(() => this.focusParagraph.style.animation = 'fadeIn 0.3s ease', 10);
        }
    }

    // Toggle play/pause in focus mode
    async focusTogglePlay() {
        if (this.focusIsPlaying && this.currentAudio) {
            // Pause
            this.currentAudio.pause();
            this.focusIsPlaying = false;
            this.isPlaying = false;
            this.updateFocusPlayButton();
        } else {
            // Play current sentence
            await this.focusPlayCurrent();
        }
    }

    // Play current sentence in focus mode
    async focusPlayCurrent() {
        const sentence = this.focusSentences[this.focusCurrentIndex];
        if (!sentence) return;

        const status = this.audioStatus[sentence.id];

        if (status === 'ready') {
            // Update return position
            this.focusReturnSentenceId = sentence.id;
            await this.playSentence(sentence.id);
            this.focusIsPlaying = true;
            this.updateFocusDisplay();
            this.updateFocusPlayButton();
        } else if (status === 'generating' || status === 'pending') {
            // Audio not ready yet - queue it
            this.pendingPlaySentenceId = sentence.id;
            this.focusReturnSentenceId = sentence.id;
            this.updateFocusDisplay();
            // If pending, try to prioritize it
            if (status === 'pending') {
                await this.prioritizeSentences([sentence.id]);
            }
        } else {
            // Audio failed or unknown status - try to skip to next ready sentence
            if (this.focusAutoplay && this.focusCurrentIndex < this.focusSentences.length - 1) {
                // Find next sentence with ready audio
                for (let i = this.focusCurrentIndex + 1; i < this.focusSentences.length; i++) {
                    if (this.audioStatus[this.focusSentences[i].id] === 'ready') {
                        this.focusCurrentIndex = i;
                        this.updateFocusDisplay();
                        setTimeout(() => this.focusPlayCurrent(), 100);
                        return;
                    }
                }
            }
        }
    }

    // Update focus mode play/pause button icon
    updateFocusPlayButton() {
        if (!this.focusPlayIcon || !this.focusPauseIcon) return;

        if (this.focusIsPlaying) {
            this.focusPlayIcon.style.display = 'none';
            this.focusPauseIcon.style.display = 'inline';
        } else {
            this.focusPlayIcon.style.display = 'inline';
            this.focusPauseIcon.style.display = 'none';
        }
    }

    // Handle audio ended specifically in focus mode
    onFocusAudioEnded() {
        this.focusIsPlaying = false;
        this.updateFocusPlayButton();
        this.updateFocusDisplay();

        // Auto-advance if autoplay is enabled
        if (this.focusAutoplay && this.focusCurrentIndex < this.focusSentences.length - 1) {
            this.focusCurrentIndex++;
            // Update return position to track where we are
            this.focusReturnSentenceId = this.focusSentences[this.focusCurrentIndex].id;
            this.updateFocusDisplay();
            // Small delay before playing next
            setTimeout(() => this.focusPlayCurrent(), 300);
        }
    }

    // Skip 5 sentences forward (like a page)
    focusNextPage() {
        const newIndex = Math.min(this.focusCurrentIndex + 5, this.focusSentences.length - 1);
        if (newIndex !== this.focusCurrentIndex) {
            this.focusCurrentIndex = newIndex;
            this.focusReturnSentenceId = this.focusSentences[this.focusCurrentIndex].id;
            this.updateFocusDisplay();
        }
    }

    // Skip 5 sentences back
    focusPrevPage() {
        const newIndex = Math.max(this.focusCurrentIndex - 5, 0);
        if (newIndex !== this.focusCurrentIndex) {
            this.focusCurrentIndex = newIndex;
            this.focusReturnSentenceId = this.focusSentences[this.focusCurrentIndex].id;
            this.updateFocusDisplay();
        }
    }

    handleKeyDown(e) {
        if (!this.focusModeActive) return;

        switch (e.key) {
            case 'ArrowUp':
                e.preventDefault();
                this.focusPrevSentence();
                break;
            case 'ArrowDown':
                e.preventDefault();
                this.focusNextSentence();
                break;
            case 'ArrowLeft':
                e.preventDefault();
                this.focusPrevPage();
                break;
            case 'ArrowRight':
                e.preventDefault();
                this.focusNextPage();
                break;
            case ' ':
                e.preventDefault();
                this.focusTogglePlay();
                break;
            case 'Escape':
                this.exitFocusMode();
                break;
        }
    }

    // =========================================================================
    // Settings & Themes
    // =========================================================================

    toggleSettings() {
        if (!this.settingsPanel) return;

        const isHidden = window.getComputedStyle(this.settingsPanel).display === 'none';

        if (isHidden) {
            this.settingsPanel.style.display = 'block';
            this.updateStatsDisplay();
        } else {
            this.settingsPanel.style.display = 'none';
        }
    }

    setTheme(theme) {
        document.body.classList.remove('theme-light', 'theme-sepia', 'theme-dark');
        if (theme !== 'light') {
            document.body.classList.add(`theme-${theme}`);
        }

        this.currentTheme = theme;
        localStorage.setItem('theme', theme);

        // Update button states
        document.querySelectorAll('.theme-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.theme === theme);
        });
    }

    toggleBionic(enabled) {
        this.bionicEnabled = enabled;
        localStorage.setItem('bionic', enabled);

        // Re-render current chapter with bionic text
        if (this.currentBook && this.chapterContent) {
            this.applyBionicToContent();
        }
    }

    applyBionicReading(text) {
        // Bold first 40-50% of each word
        return text.replace(/\b(\w+)\b/g, (match) => {
            if (match.length <= 2) return `<b>${match}</b>`;
            const boldLen = Math.ceil(match.length * 0.4);
            return `<b>${match.slice(0, boldLen)}</b>${match.slice(boldLen)}`;
        });
    }

    applyBionicToContent() {
        if (!this.bionicEnabled) {
            // Remove bionic formatting
            const chapter = this.currentBook.chapters[this.currentChapter];
            if (chapter) {
                this.chapterContent.innerHTML = chapter.html || '';
                this.attachSentenceHandlers();
                this.applyNotesHighlights();
            }
            return;
        }

        // Apply bionic to all text nodes
        const walker = document.createTreeWalker(
            this.chapterContent,
            NodeFilter.SHOW_TEXT,
            null,
            false
        );

        const textNodes = [];
        while (walker.nextNode()) {
            textNodes.push(walker.currentNode);
        }

        textNodes.forEach(node => {
            if (node.textContent.trim()) {
                const span = document.createElement('span');
                span.className = 'bionic-text';
                span.innerHTML = this.applyBionicReading(node.textContent);
                node.parentNode.replaceChild(span, node);
            }
        });
    }

    applySettings() {
        // Apply theme
        this.setTheme(this.currentTheme);

        // Apply bionic toggle state
        this.bionicToggle.checked = this.bionicEnabled;

        // Update stats display
        this.updateStatsDisplay();
    }

    // =========================================================================
    // Session Timer
    // =========================================================================

    startSession(minutes) {
        if (this.sessionActive) {
            this.endSession();
        }

        this.sessionTimeLeft = minutes * 60;
        this.sessionActive = true;
        this.sessionTimer.style.display = 'block';
        this.updateSessionDisplay();

        // Close settings panel
        this.settingsPanel.style.display = 'none';

        this.sessionInterval = setInterval(() => {
            this.sessionTimeLeft--;
            this.updateSessionDisplay();

            if (this.sessionTimeLeft <= 0) {
                this.completeSession(minutes);
            }
        }, 1000);
    }

    updateSessionDisplay() {
        const mins = Math.floor(this.sessionTimeLeft / 60);
        const secs = this.sessionTimeLeft % 60;
        this.sessionTimeLeftEl.textContent = `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    completeSession(minutes) {
        this.endSession();
        this.showCelebration('Session Complete!', `You focused for ${minutes} minutes!`);
        this.updateStats('session', minutes);
    }

    endSession() {
        this.sessionActive = false;
        this.sessionTimer.style.display = 'none';
        if (this.sessionInterval) {
            clearInterval(this.sessionInterval);
            this.sessionInterval = null;
        }
    }

    // =========================================================================
    // Stats & Streaks
    // =========================================================================

    loadStats() {
        const saved = localStorage.getItem('reader_stats');
        if (saved) {
            return JSON.parse(saved);
        }
        return {
            streak: 0,
            lastReadDate: null,
            todayMinutes: 0,
            totalPages: 0,
            todayDate: null
        };
    }

    saveStats() {
        localStorage.setItem('reader_stats', JSON.stringify(this.stats));
    }

    updateStats(type, value = 0) {
        const today = new Date().toDateString();

        // Check if it's a new day
        if (this.stats.todayDate !== today) {
            // Check streak
            const lastDate = this.stats.lastReadDate ? new Date(this.stats.lastReadDate) : null;
            const todayDate = new Date(today);

            if (lastDate) {
                const diffDays = Math.floor((todayDate - lastDate) / (1000 * 60 * 60 * 24));
                if (diffDays === 1) {
                    this.stats.streak++;
                } else if (diffDays > 1) {
                    this.stats.streak = 1;
                }
            } else {
                this.stats.streak = 1;
            }

            this.stats.todayMinutes = 0;
            this.stats.todayDate = today;
        }

        this.stats.lastReadDate = today;

        if (type === 'session') {
            this.stats.todayMinutes += value;
        } else if (type === 'chapter') {
            this.stats.totalPages += 10; // Approximate pages per chapter
        }

        this.saveStats();
        this.updateStatsDisplay();
    }

    updateStatsDisplay() {
        if (this.statStreak) this.statStreak.textContent = this.stats.streak || 0;
        if (this.statToday) this.statToday.textContent = this.stats.todayMinutes || 0;
        if (this.statTotal) this.statTotal.textContent = this.stats.totalPages || 0;
    }

    // =========================================================================
    // Celebrations
    // =========================================================================

    showCelebration(text, subtext) {
        const celebrationText = this.celebration.querySelector('.celebration-text');
        const celebrationSubtext = this.celebration.querySelector('.celebration-subtext');

        celebrationText.textContent = text;
        celebrationSubtext.textContent = subtext;

        this.celebration.style.display = 'flex';

        // Auto-hide after 3 seconds
        setTimeout(() => {
            this.celebration.style.display = 'none';
        }, 3000);

        // Click to dismiss
        this.celebration.onclick = () => {
            this.celebration.style.display = 'none';
        };
    }

    // =========================================================================
    // Library Management
    // =========================================================================

    async loadLibrary() {
        try {
            const response = await fetch('/library');
            const data = await response.json();

            if (data.books && data.books.length > 0) {
                this.renderLibrary(data.books);
            }
        } catch (error) {
            console.error('Error loading library:', error);
        }
    }

    renderLibrary(books) {
        if (!this.librarySection || !this.libraryGrid) return;

        if (books.length === 0) {
            this.librarySection.style.display = 'none';
            return;
        }

        this.librarySection.style.display = 'block';
        this.libraryGrid.innerHTML = '';

        books.forEach(book => {
            const card = this.createBookCard(book);
            this.libraryGrid.appendChild(card);
        });
    }

    createBookCard(book) {
        const card = document.createElement('div');
        card.className = 'book-card';
        card.dataset.bookId = book.book_id;

        const readingProgress = book.reading_progress?.percentage || 0;
        const audioProgress = book.audio_progress?.percentage || 0;

        // Generate placeholder color based on book title
        const hue = this.hashStringToHue(book.title);

        card.innerHTML = `
            <div class="book-cover" style="background: linear-gradient(135deg, hsl(${hue}, 60%, 50%) 0%, hsl(${hue + 30}, 50%, 40%) 100%);">
                ${book.cover_url
                    ? `<img src="${book.cover_url}" alt="${book.title}" onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
                       <div class="book-cover-placeholder" style="display: none;">
                           <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                               <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>
                               <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>
                           </svg>
                           <span class="placeholder-title">${book.title}</span>
                       </div>`
                    : `<div class="book-cover-placeholder">
                           <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                               <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>
                               <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>
                           </svg>
                           <span class="placeholder-title">${book.title}</span>
                       </div>`
                }
                <button class="book-delete-btn" title="Remove from library">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="18" y1="6" x2="6" y2="18"/>
                        <line x1="6" y1="6" x2="18" y2="18"/>
                    </svg>
                </button>
            </div>
            <div class="book-info">
                <div class="book-title-text">${book.title}</div>
                ${book.author ? `<div class="book-author">${book.author}</div>` : ''}
                <div class="book-progress">
                    <div class="progress-row">
                        <span class="progress-icon">📖</span>
                        <div class="progress-bar-wrapper">
                            <div class="progress-bar-fill reading" style="width: ${readingProgress}%"></div>
                        </div>
                        <span class="progress-text">${Math.round(readingProgress)}%</span>
                    </div>
                    <div class="progress-row">
                        <span class="progress-icon">🔊</span>
                        <div class="progress-bar-wrapper">
                            <div class="progress-bar-fill audio" style="width: ${audioProgress}%"></div>
                        </div>
                        <span class="progress-text">${Math.round(audioProgress)}%</span>
                    </div>
                </div>
            </div>
        `;

        // Click to open book
        card.addEventListener('click', (e) => {
            if (e.target.closest('.book-delete-btn')) {
                e.stopPropagation();
                this.removeFromLibrary(book.book_id, book.title);
                return;
            }
            this.openBookFromLibrary(book.book_id);
        });

        return card;
    }

    hashStringToHue(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            hash = str.charCodeAt(i) + ((hash << 5) - hash);
        }
        return Math.abs(hash) % 360;
    }

    async removeFromLibrary(bookId, title) {
        if (!confirm(`Remove "${title}" from your library?`)) return;

        try {
            await fetch(`/library/${bookId}`, { method: 'DELETE' });
            // Reload library
            await this.loadLibrary();
            this.showToast('Book removed from library');
        } catch (error) {
            console.error('Error removing book:', error);
        }
    }

    async openBookFromLibrary(bookId) {
        this.showLoading('Opening book...');

        try {
            // Load book from server (will re-process if needed)
            const response = await fetch(`/load_book/${bookId}`, { method: 'POST' });
            const data = await response.json();

            if (data.success) {
                await this.loadBookContent();
            } else {
                this.showToast(data.error || 'Book file not found. Please re-upload.');
            }
        } catch (error) {
            console.error('Error opening book from library:', error);
            this.showToast('Error opening book');
        } finally {
            this.hideLoading();
        }
    }

    async updateLibraryProgress() {
        if (!this.currentBook) return;

        try {
            await fetch('/library/update_progress', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    book_id: this.currentBook.book_id,
                    chapter_index: this.currentChapter,
                    sentence_index: this.currentSentence?.sentence_index || 0,
                    total_chapters: this.currentBook.chapters?.length || 0
                })
            });
        } catch (error) {
            console.error('Error updating library progress:', error);
        }
    }
}

// Initialize (single instance)
document.addEventListener('DOMContentLoaded', () => {
    window.reader = new ProgressiveEPUBReader();
});
