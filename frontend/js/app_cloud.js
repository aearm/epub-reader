/**
 * EPUB Reader - Cloud Version
 *
 * Integrates with distributed audio coordinator for:
 * - Streaming audio from S3
 * - Volunteer mode for generating audio
 * - Cross-device sync
 * - Wake lock to prevent screen sleep
 */

class EPUBReaderCloud {
    constructor() {
        // Core state
        this.currentBook = null;
        this.sentences = [];
        this.currentChapter = 0;
        this.currentSentence = null;
        this.isPlaying = false;

        // Audio
        this.audioElements = {};
        this.audioStatus = {};
        this.playbackSpeed = 1.0;
        this.autoplay = true;  // Default to true for continuous reading
        this.currentAudio = null;
        this.currentEndedHandler = null;
        this.playbackRequestId = 0;
        this.prefetchTimer = null;
        this.sentenceById = new Map();
        this.generationPromises = new Map();
        this.maxBackgroundGenerations = 2;
        this.lastPlayWasSingleShot = false;
        this.playbackQueue = [];
        this.currentBookId = null;
        this.libraryBooks = [];
        this.progressSyncTimer = null;
        this.bookSentenceCache = new Map();
        this.localAudioBlobUrls = {};
        this.bookActionState = new Map();
        this.audioDbPromise = null;
        this.librarySearchQuery = '';

        // Features
        this.notes = {};
        this.focusModeActive = false;
        this.volunteerMode = false;

        // Wake Lock
        this.wakeLock = null;
        this._expectSpeechInterruption = false;

        // Initialize
        this.initializeConfig();
        this.initializeAuth();
        this.initializeCoordinator();
        this.initializeElements();
        this.attachEventListeners();
        this.handleSidebarResponsive();
        this.initAudioCache();
        this.loadSettings();
        this.checkCoordinatorStatus();
        if (this.authClient?.isAuthenticated()) {
            this.loadLibrary();
        }
    }

    // =========================================================================
    // Initialization
    // =========================================================================

    initializeConfig() {
        this.config = window.EPUB_READER_CONFIG || {
            apiUrl: 'https://api.reader.psybytes.com',
            localWorkerUrl: 'http://127.0.0.1:5001',
            cognito: {
                userPoolId: '',
                clientId: '',
                domain: '',
                region: 'eu-west-1'
            }
        };
    }

    initializeAuth() {
        const cognitoConfig = this.config.cognito || {};

        if (!cognitoConfig.clientId) {
            console.warn('Cognito not configured, running in offline mode');
            this.authClient = null;
            // Still update UI to show state
            setTimeout(() => this.updateAuthUI(), 100);
            return;
        }

        console.log('Initializing Cognito auth with:', cognitoConfig.domain);

        this.authClient = new AuthClient({
            userPoolId: cognitoConfig.userPoolId,
            clientId: cognitoConfig.clientId,
            domain: cognitoConfig.domain,
            region: cognitoConfig.region || 'eu-west-1',
            redirectUri: window.location.origin,
            onAuthChange: (user) => this.onAuthChange(user)
        });

        // Handle OAuth callback
        if (window.location.search.includes('code=')) {
            this.authClient.handleCallback();
        }

        // Update UI after a short delay to ensure DOM is ready
        setTimeout(() => this.updateAuthUI(), 100);
    }

    initializeCoordinator() {
        this.coordinator = new CoordinatorClient({
            apiUrl: this.config.apiUrl,
            onAuthRequired: () => {
                if (this.authClient) {
                    this.authClient.login();
                }
            }
        });

        // Set token if authenticated
        if (this.authClient && this.authClient.accessToken) {
            this.coordinator.setAccessToken(this.authClient.accessToken);
        }
    }

    initializeElements() {
        // Auth UI
        this.loginBtn = document.getElementById('loginBtn');
        this.logoutBtn = document.getElementById('logoutBtn');
        this.userInfo = document.getElementById('userInfo');
        this.userAvatar = document.getElementById('userAvatar');
        this.userName = document.getElementById('userName');
        this.welcomeLoginBtn = document.getElementById('welcomeLoginBtn');
        this.authPrompt = document.getElementById('authPrompt');
        this.libraryHomeShell = document.getElementById('libraryHomeShell');
        this.librarySection = document.getElementById('librarySection');
        this.libraryGrid = document.getElementById('libraryGrid');
        this.librarySearchInput = document.getElementById('librarySearchInput');
        this.homeImportBtn = document.getElementById('homeImportBtn');
        this.continueReadingSection = document.getElementById('continueReadingSection');
        this.continueReadingGrid = document.getElementById('continueReadingGrid');

        // Main UI
        this.welcomeScreen = document.getElementById('welcomeScreen');
        this.readerContainer = document.getElementById('readerContainer');
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.loadingText = document.getElementById('loadingText');
        this.headerTitle = document.getElementById('headerTitle');

        // File handling
        this.fileInput = document.getElementById('fileInput');
        this.uploadBtn = document.getElementById('uploadBtn');
        this.clearBtn = document.getElementById('clearBtn');

        // Book content
        this.bookTitle = document.getElementById('bookTitle');
        this.chapterContent = document.getElementById('chapterContent');
        this.chapterList = document.getElementById('chapterList');

        // Audio controls
        this.audioControls = document.getElementById('audioControls');
        this.playPauseBtn = document.getElementById('playPauseBtn');
        this.prevBtn = document.getElementById('prevBtn');
        this.nextBtn = document.getElementById('nextBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.speedSlider = document.getElementById('speedSlider');
        this.speedValue = document.getElementById('speedValue');
        this.autoplayCheckbox = document.getElementById('autoplayCheckbox');

        // Progress
        this.audioReady = document.getElementById('audioReady');
        this.audioTotal = document.getElementById('audioTotal');
        this.audioProgressBar = document.getElementById('audioProgressBar');

        // Settings
        this.settingsPanel = document.getElementById('settingsPanel');
        this.toggleSettings = document.getElementById('toggleSettings');
        this.volunteerToggle = document.getElementById('volunteerToggle');

        // Sidebars
        this.leftSidebar = document.getElementById('leftSidebar');
        this.toggleChapters = document.getElementById('toggleChapters');
        this.sidebarBackdrop = document.getElementById('sidebarBackdrop');

        // Focus mode
        this.focusMode = document.getElementById('focusMode');
        this.toggleFocus = document.getElementById('toggleFocus');

        // Coordinator
        this.coordinatorStats = document.getElementById('coordinatorStats');
        this.coordinatorDot = document.getElementById('coordinatorDot');
        this.coordinatorStatusText = document.getElementById('coordinatorStatusText');

        // Wake lock indicator
        this.wakeLockIndicator = document.getElementById('wakeLockIndicator');
    }

    attachEventListeners() {
        // Auth
        this.loginBtn?.addEventListener('click', () => this.authClient?.login());
        this.logoutBtn?.addEventListener('click', () => this.authClient?.logout());
        this.welcomeLoginBtn?.addEventListener('click', () => this.authClient?.login());

        // File handling
        this.uploadBtn?.addEventListener('click', () => this.fileInput.click());
        this.homeImportBtn?.addEventListener('click', () => this.fileInput.click());
        this.fileInput?.addEventListener('change', (e) => this.handleFileUpload(e));
        this.clearBtn?.addEventListener('click', () => this.clearBook());
        this.librarySearchInput?.addEventListener('input', (e) => {
            this.librarySearchQuery = (e.target.value || '').trim().toLowerCase();
            this.renderLibrary(this.libraryBooks);
        });

        // Audio controls
        this.playPauseBtn?.addEventListener('click', () => this.togglePlayPause());
        this.prevBtn?.addEventListener('click', () => this.playPrevSentence());
        this.nextBtn?.addEventListener('click', () => this.playNextSentence());
        this.stopBtn?.addEventListener('click', () => this.stopPlayback());
        this.speedSlider?.addEventListener('input', (e) => this.setPlaybackSpeed(e.target.value));
        this.autoplayCheckbox?.addEventListener('change', (e) => {
            this.autoplay = e.target.checked;
            localStorage.setItem('epub_autoplay', this.autoplay);
        });

        // Settings
        this.toggleSettings?.addEventListener('click', () => this.toggleSettingsPanel());
        this.volunteerToggle?.addEventListener('change', (e) => this.setVolunteerMode(e.target.checked));
        document.getElementById('closeSettings')?.addEventListener('click', () => this.toggleSettingsPanel());

        // Sidebar
        this.toggleChapters?.addEventListener('click', () => this.toggleSidebar());

        // Focus mode
        this.toggleFocus?.addEventListener('click', () => this.toggleFocusMode());

        // Theme buttons
        document.querySelectorAll('.theme-btn').forEach(btn => {
            btn.addEventListener('click', () => this.setTheme(btn.dataset.theme));
        });

        // Visibility change - manage wake lock
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.releaseWakeLock();
            } else if (this.isPlaying) {
                this.requestWakeLock();
            }
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this.handleKeyboard(e));
        window.addEventListener('scroll', () => this.scheduleVisiblePrefetch(), { passive: true });
        window.addEventListener('resize', () => this.scheduleVisiblePrefetch());
        this.chapterContent?.addEventListener('scroll', () => this.scheduleVisiblePrefetch(), { passive: true });
        this.sidebarBackdrop?.addEventListener('click', () => this.closeSidebar());
        window.addEventListener('resize', () => this.handleSidebarResponsive());
    }

    // =========================================================================
    // Authentication
    // =========================================================================

    onAuthChange(user) {
        this.updateAuthUI();

        if (user && this.authClient) {
            this.coordinator.setAccessToken(this.authClient.accessToken);
            this.checkCoordinatorStatus();
            this.loadLibrary();
        } else {
            this.libraryBooks = [];
            if (this.libraryHomeShell) this.libraryHomeShell.style.display = 'none';
            if (this.librarySection) this.librarySection.style.display = 'none';
            if (this.welcomeScreen) this.welcomeScreen.classList.remove('library-mode');
            document.body.classList.remove('home-shell-active');
        }
    }

    updateAuthUI() {
        const isLoggedIn = this.authClient?.isAuthenticated() || false;
        const user = this.authClient?.getUser();

        console.log('Auth UI update - logged in:', isLoggedIn, 'user:', user);

        // Get elements fresh in case they weren't ready before
        const loginBtn = document.getElementById('loginBtn');
        const userInfo = document.getElementById('userInfo');
        const authPrompt = document.getElementById('authPrompt');
        const userAvatar = document.getElementById('userAvatar');
        const userName = document.getElementById('userName');

        // Show login button if not logged in
        if (loginBtn) {
            loginBtn.style.display = isLoggedIn ? 'none' : 'inline-block';
            console.log('Login button display:', loginBtn.style.display);
        }

        if (userInfo) {
            userInfo.style.display = isLoggedIn ? 'flex' : 'none';
        }

        if (authPrompt) {
            authPrompt.style.display = isLoggedIn ? 'none' : 'block';
        }

        if (isLoggedIn && user) {
            if (userAvatar) userAvatar.textContent = (user.name || user.email || 'U')[0].toUpperCase();
            if (userName) userName.textContent = user.name || user.email;
        }
    }

    // =========================================================================
    // Coordinator
    // =========================================================================

    async checkCoordinatorStatus() {
        try {
            await this.coordinator.getStats();

            if (this.coordinatorDot) this.coordinatorDot.classList.add('connected');
            if (this.coordinatorStatusText) this.coordinatorStatusText.textContent = 'Connected';
            if (this.coordinatorStats) this.coordinatorStats.style.display = 'block';
        } catch (error) {
            console.warn('Coordinator not available:', error);
            if (this.coordinatorDot) this.coordinatorDot.classList.remove('connected');
            if (this.coordinatorStatusText) this.coordinatorStatusText.textContent = 'Offline mode';
        }
    }

    initAudioCache() {
        if (!('indexedDB' in window)) {
            this.audioDbPromise = Promise.resolve(null);
            return;
        }

        this.audioDbPromise = new Promise((resolve) => {
            const request = indexedDB.open('epub_reader_audio_cache', 2);

            request.onupgradeneeded = () => {
                const db = request.result;
                if (!db.objectStoreNames.contains('audio_files')) {
                    db.createObjectStore('audio_files', { keyPath: 'id' });
                }
                if (!db.objectStoreNames.contains('book_files')) {
                    db.createObjectStore('book_files', { keyPath: 'book_id' });
                }
            };

            request.onsuccess = () => resolve(request.result);
            request.onerror = () => {
                console.warn('IndexedDB unavailable for audio cache');
                resolve(null);
            };
        });
    }

    async getAudioDb() {
        if (!this.audioDbPromise) {
            this.initAudioCache();
        }
        return this.audioDbPromise;
    }

    makeAudioCacheId(bookId, key, kind = 'hash') {
        return `${bookId || 'global'}:${kind}:${key}`;
    }

    async getCachedAudioBlob(bookId, key, kind = 'hash') {
        const db = await this.getAudioDb();
        if (!db) return null;

        return new Promise((resolve) => {
            const tx = db.transaction('audio_files', 'readonly');
            const store = tx.objectStore('audio_files');
            const req = store.get(this.makeAudioCacheId(bookId, key, kind));

            req.onsuccess = () => resolve(req.result?.blob || null);
            req.onerror = () => resolve(null);
        });
    }

    async setCachedAudioBlob(bookId, key, blob, kind = 'hash') {
        if (!blob) return;
        const db = await this.getAudioDb();
        if (!db) return;

        await new Promise((resolve) => {
            const tx = db.transaction('audio_files', 'readwrite');
            const store = tx.objectStore('audio_files');
            store.put({
                id: this.makeAudioCacheId(bookId, key, kind),
                book_id: bookId || '',
                key,
                kind,
                blob,
                mime_type: blob.type || 'audio/mpeg',
                updated_at: Date.now()
            });
            tx.oncomplete = () => resolve();
            tx.onerror = () => resolve();
        });
    }

    getOrCreateBlobUrl(cacheId, blob) {
        if (this.localAudioBlobUrls[cacheId]) {
            return this.localAudioBlobUrls[cacheId];
        }
        const url = URL.createObjectURL(blob);
        this.localAudioBlobUrls[cacheId] = url;
        return url;
    }

    async getCachedBookBlob(bookId) {
        if (!bookId) return null;
        const db = await this.getAudioDb();
        if (!db || !db.objectStoreNames.contains('book_files')) return null;

        return new Promise((resolve) => {
            const tx = db.transaction('book_files', 'readonly');
            const store = tx.objectStore('book_files');
            const req = store.get(bookId);
            req.onsuccess = () => resolve(req.result?.blob || null);
            req.onerror = () => resolve(null);
        });
    }

    async setCachedBookBlob(bookId, blob, filename = 'book.epub') {
        if (!bookId || !blob) return;
        const db = await this.getAudioDb();
        if (!db || !db.objectStoreNames.contains('book_files')) return;

        await new Promise((resolve) => {
            const tx = db.transaction('book_files', 'readwrite');
            const store = tx.objectStore('book_files');
            store.put({
                book_id: bookId,
                blob,
                filename,
                updated_at: Date.now()
            });
            tx.oncomplete = () => resolve();
            tx.onerror = () => resolve();
        });
    }

    // =========================================================================
    // Wake Lock (Prevent Screen Sleep)
    // =========================================================================

    async requestWakeLock() {
        if (!('wakeLock' in navigator)) {
            console.log('Wake Lock API not supported');
            return;
        }

        try {
            this.wakeLock = await navigator.wakeLock.request('screen');
            console.log('Wake lock acquired');

            if (this.wakeLockIndicator) {
                this.wakeLockIndicator.classList.add('active');
                setTimeout(() => {
                    this.wakeLockIndicator.classList.remove('active');
                }, 3000);
            }

            this.wakeLock.addEventListener('release', () => {
                console.log('Wake lock released');
            });
        } catch (err) {
            console.warn('Wake lock request failed:', err);
        }
    }

    releaseWakeLock() {
        if (this.wakeLock) {
            this.wakeLock.release();
            this.wakeLock = null;
        }
    }

    cancelBrowserSpeech(expected = true) {
        if (!('speechSynthesis' in window)) return;

        if (expected) {
            this._expectSpeechInterruption = true;
        }
        speechSynthesis.cancel();

        if (expected) {
            setTimeout(() => {
                this._expectSpeechInterruption = false;
            }, 100);
        }
    }

    // =========================================================================
    // File Handling
    // =========================================================================

    async computeBookId(file) {
        const buffer = await file.arrayBuffer();
        const digest = await crypto.subtle.digest('SHA-256', buffer);
        const bytes = new Uint8Array(digest);
        const hex = Array.from(bytes).map(b => b.toString(16).padStart(2, '0')).join('');
        return hex.slice(0, 32);
    }

    dataUrlToBlob(dataUrl) {
        const parts = String(dataUrl || '').split(',');
        if (parts.length !== 2) return null;
        const mime = (parts[0].match(/data:(.*?);base64/) || [])[1] || 'application/octet-stream';
        const binary = atob(parts[1]);
        const array = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {
            array[i] = binary.charCodeAt(i);
        }
        return new Blob([array], { type: mime });
    }

    async handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        this.showLoading('Processing book...');

        try {
            const bookId = await this.computeBookId(file);
            const book = await this.parseEPUB(file, { bookId, sourceName: file.name });
            this.currentBook = book;
            this.sentences = book.sentences;
            this.currentBookId = book.book_id;
            await this.setCachedBookBlob(book.book_id, file, file.name || `${book.book_id}.epub`);

            if (this.authClient?.isAuthenticated()) {
                try {
                    await this.uploadBookToCloud(file, book);
                    await this.loadLibrary();
                } catch (cloudError) {
                    console.error('Cloud upload skipped:', cloudError);
                    alert('Book opened locally. Cloud sync is temporarily unavailable.');
                }
            }

            this.displayBook();
            this.applySavedProgress();
            this.prefetchAudio();
            this.scheduleProgressSync();
        } catch (error) {
            console.error('Error processing EPUB:', error);
            alert('Failed to process EPUB file');
        } finally {
            this.hideLoading();
        }
    }

    async parseEPUB(file, options = {}) {
        const JSZip = window.JSZip;
        if (!JSZip) {
            throw new Error('JSZip not loaded');
        }

        const zip = await JSZip.loadAsync(file);
        const sourceName = options.sourceName || file?.name || 'book.epub';

        // Find container.xml to get content location
        const containerXml = await zip.file('META-INF/container.xml')?.async('text');
        if (!containerXml) {
            throw new Error('Invalid EPUB: missing container.xml');
        }

        // Parse container to find content.opf path
        const parser = new DOMParser();
        const containerDoc = parser.parseFromString(containerXml, 'text/xml');
        const rootfile = containerDoc.querySelector('rootfile');
        const contentPath = rootfile?.getAttribute('full-path') || 'OEBPS/content.opf';
        const basePath = contentPath.substring(0, contentPath.lastIndexOf('/') + 1);

        // Load content.opf
        const contentOpf = await zip.file(contentPath)?.async('text');
        if (!contentOpf) {
            throw new Error('Invalid EPUB: missing content.opf');
        }

        const opfDoc = parser.parseFromString(contentOpf, 'text/xml');

        // Get metadata
        const titleEl = opfDoc.querySelector('metadata title, dc\\:title');
        const creatorEl = opfDoc.querySelector('metadata creator, dc\\:creator');

        const coverDataUrl = await this.extractCoverDataUrl(zip, opfDoc, basePath);

        const result = {
            book_id: options.bookId || null,
            title: titleEl?.textContent || sourceName.replace('.epub', ''),
            author: creatorEl?.textContent || '',
            cover_data_url: coverDataUrl,
            chapters: [],
            sentences: []
        };

        // Get spine items
        const spine = opfDoc.querySelectorAll('spine itemref');
        const manifest = opfDoc.querySelectorAll('manifest item');
        const manifestMap = {};
        manifest.forEach(item => {
            manifestMap[item.getAttribute('id')] = item.getAttribute('href');
        });

        let sentenceIndex = 0;

        // Process each spine item (chapter)
        for (let i = 0; i < spine.length; i++) {
            const idref = spine[i].getAttribute('idref');
            const href = manifestMap[idref];
            if (!href) continue;

            const chapterPath = basePath + href;
            const chapterContent = await zip.file(chapterPath)?.async('text');
            if (!chapterContent) continue;

            const chapterDoc = parser.parseFromString(chapterContent, 'text/html');
            const body = chapterDoc.body;
            if (!body) continue;

            // Get chapter title
            const titleTag = chapterDoc.querySelector('h1, h2, h3, title');
            const chapterTitle = titleTag?.textContent?.trim() || `Chapter ${i + 1}`;

            // Process images - convert to data URLs
            const images = body.querySelectorAll('img');
            for (const img of images) {
                const src = img.getAttribute('src');
                if (src) {
                    const imgPath = basePath + src.replace('../', '');
                    const imgData = await zip.file(imgPath)?.async('base64');
                    if (imgData) {
                        const ext = src.split('.').pop().toLowerCase();
                        const mime = ext === 'png' ? 'image/png' : 'image/jpeg';
                        img.setAttribute('src', `data:${mime};base64,${imgData}`);
                    }
                }
            }

            // Extract sentences
            const text = body.textContent || '';
            const chapterSentences = [];

            // Improved sentence splitting with abbreviation awareness
            const sentences = this.splitIntoSentences(text);
            for (const sentenceText of sentences) {
                const cleaned = sentenceText.trim().replace(/\s+/g, ' ');
                if (cleaned.length > 5 && /[a-zA-Z]/.test(cleaned)) {
                    const sentence = {
                        id: `s_${sentenceIndex}`,
                        text: cleaned,
                        chapterIndex: i,
                        sentence_index: sentenceIndex
                    };
                    chapterSentences.push(sentence);
                    result.sentences.push(sentence);
                    sentenceIndex++;
                }
            }

            // Wrap sentences in spans for highlighting
            // We'll do this simpler - just mark paragraphs with sentence data
            let html = body.innerHTML;

            result.chapters.push({
                index: i,
                title: chapterTitle,
                html: html,
                sentences: chapterSentences
            });
        }

        console.log(`Parsed EPUB: ${result.chapters.length} chapters, ${result.sentences.length} sentences`);
        return result;
    }

    async extractCoverDataUrl(zip, opfDoc, basePath) {
        const normalizePath = (p) => p.replace(/^\.?\//, '').replace(/^\.\.\//, '');

        let coverHref = null;

        const metaCover = opfDoc.querySelector('metadata meta[name="cover"]');
        const coverId = metaCover?.getAttribute('content');
        if (coverId) {
            const coverItem = opfDoc.querySelector(`manifest item[id="${coverId}"]`);
            coverHref = coverItem?.getAttribute('href') || null;
        }

        if (!coverHref) {
            const coverItem = opfDoc.querySelector('manifest item[properties*="cover-image"]');
            coverHref = coverItem?.getAttribute('href') || null;
        }

        if (!coverHref) {
            const firstImage = opfDoc.querySelector('manifest item[media-type^="image/"]');
            coverHref = firstImage?.getAttribute('href') || null;
        }

        if (!coverHref) return null;

        const candidatePath = normalizePath(`${basePath}${coverHref}`);
        const entry = zip.file(candidatePath) || zip.file(normalizePath(coverHref));
        if (!entry) return null;

        const bytes = await entry.async('uint8array');
        const ext = (coverHref.split('.').pop() || '').toLowerCase();
        const mime = ext === 'png' ? 'image/png' : (ext === 'webp' ? 'image/webp' : 'image/jpeg');

        let binary = '';
        const chunkSize = 0x8000;
        for (let i = 0; i < bytes.length; i += chunkSize) {
            binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
        }
        return `data:${mime};base64,${btoa(binary)}`;
    }

    /**
     * Split text into sentences with abbreviation awareness
     */
    splitIntoSentences(text) {
        // Prefer native sentence segmentation when available.
        if (typeof Intl !== 'undefined' && typeof Intl.Segmenter === 'function') {
            try {
                const normalized = text.replace(/\s+/g, ' ').trim();
                const segmenter = new Intl.Segmenter('en', { granularity: 'sentence' });
                const segments = Array.from(segmenter.segment(normalized))
                    .map(item => (item.segment || '').trim())
                    .filter(Boolean);
                if (segments.length) {
                    return segments;
                }
            } catch (error) {
                console.warn('Intl.Segmenter sentence split failed, using fallback logic:', error);
            }
        }

        // Common abbreviations that shouldn't end sentences
        const abbreviations = new Set([
            'Mr', 'Mrs', 'Ms', 'Dr', 'Prof', 'Sr', 'Jr', 'vs', 'etc', 'Inc', 'Ltd', 'Corp',
            'St', 'Ave', 'Blvd', 'Rd', 'Gen', 'Col', 'Lt', 'Sgt', 'Rev', 'Hon',
            'Fig', 'Vol', 'No', 'Ch', 'Pt', 'pp', 'ed', 'trans', 'approx',
            'e.g', 'i.e', 'viz', 'cf', 'al'
        ]);

        // Normalize whitespace
        text = text.replace(/\s+/g, ' ').trim();

        const sentences = [];
        let currentSentence = '';
        let i = 0;

        while (i < text.length) {
            const char = text[i];
            currentSentence += char;

            // Check for sentence-ending punctuation
            if (char === '.' || char === '!' || char === '?') {
                // Look ahead - need space or end after punctuation
                const nextChar = text[i + 1];
                const isEndOfText = i === text.length - 1;
                const hasSpaceAfter = nextChar === ' ' || nextChar === '\n';

                if (isEndOfText || hasSpaceAfter) {
                    // Check if this is an abbreviation
                    let isAbbreviation = false;

                    if (char === '.') {
                        // Extract word before the period
                        const words = currentSentence.trim().split(/\s+/);
                        const lastWord = words[words.length - 1].replace(/\.$/, '');

                        // Check if it's an abbreviation
                        if (abbreviations.has(lastWord)) {
                            isAbbreviation = true;
                        }

                        // Check for single letter (initials like "J.")
                        if (lastWord.length === 1 && /[A-Z]/.test(lastWord)) {
                            isAbbreviation = true;
                        }

                        // Check for numbered items like "1." "2."
                        if (/^\d+$/.test(lastWord)) {
                            isAbbreviation = true;
                        }
                    }

                    if (!isAbbreviation) {
                        // Check if next word starts with uppercase (indicates new sentence)
                        let j = i + 1;
                        while (j < text.length && text[j] === ' ') j++;
                        const nextWordStart = text[j];

                        // End sentence if next starts with uppercase or is end of text
                        if (isEndOfText || (nextWordStart && /[A-Z"'\u201C]/.test(nextWordStart))) {
                            const trimmed = currentSentence.trim();
                            if (trimmed.length > 0) {
                                sentences.push(trimmed);
                            }
                            currentSentence = '';
                        }
                    }
                }
            }
            i++;
        }

        // Add any remaining text as last sentence
        const remaining = currentSentence.trim();
        if (remaining.length > 0) {
            sentences.push(remaining);
        }

        return sentences;
    }

    escapeHtml(text) {
        return String(text)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    renderChapterSentenceFallback(chapter) {
        const groups = [];
        const chunkSize = 4;
        for (let i = 0; i < chapter.sentences.length; i += chunkSize) {
            const chunk = chapter.sentences.slice(i, i + chunkSize);
            const line = chunk
                .map(s => `<span class="sentence" data-sentence-id="${s.id}">${this.escapeHtml(s.text)}</span>`)
                .join(' ');
            groups.push(`<p>${line}</p>`);
        }
        return groups.join('');
    }

    chapterSentenceAnchorsAreReliable(chapter) {
        const elements = Array.from(this.chapterContent.querySelectorAll('.sentence[data-sentence-id]'));
        if (!elements.length) return false;

        const renderedUnique = [];
        const seen = new Set();
        elements.forEach(el => {
            const id = el.dataset.sentenceId;
            if (!id || seen.has(id)) return;
            seen.add(id);
            renderedUnique.push(id);
        });

        const expected = chapter.sentences.map(s => s.id);
        if (renderedUnique.length !== expected.length) return false;

        for (let i = 0; i < expected.length; i++) {
            if (renderedUnique[i] !== expected[i]) {
                return false;
            }
        }
        return true;
    }

    displayBook() {
        if (!this.currentBook) return;
        this.currentBookId = this.currentBook.book_id || this.currentBookId;

        this.welcomeScreen.style.display = 'none';
        this.readerContainer.style.display = 'flex';
        this.audioControls.style.display = 'flex';
        this.clearBtn.style.display = 'inline-block';
        document.body.classList.remove('home-shell-active');
        this.sentenceById = new Map(this.sentences.map(s => [s.id, s]));

        this.headerTitle.textContent = this.currentBook.title;
        this.bookTitle.textContent = this.currentBook.title;

        this.renderChapterList();
        this.displayChapter(0);

        // Request wake lock when reading
        this.requestWakeLock();
    }

    renderChapterList() {
        if (!this.chapterList) return;

        this.chapterList.innerHTML = '';
        this.currentBook.chapters.forEach((chapter, index) => {
            const li = document.createElement('li');
            li.className = 'chapter-item' + (index === this.currentChapter ? ' active' : '');
            li.textContent = chapter.title;
            li.addEventListener('click', () => this.displayChapter(index));
            this.chapterList.appendChild(li);
        });
    }

    displayChapter(index) {
        if (!this.currentBook || index < 0 || index >= this.currentBook.chapters.length) return;

        this.currentChapter = index;
        this.playbackQueue = [];
        const chapter = this.currentBook.chapters[index];

        if (this.chapterContent) {
            // Render HTML with sentence wrapping
            let html = chapter.html || '';

            // Wrap each sentence in a span
            chapter.sentences.forEach(s => {
                // Find the first 30 chars of the sentence in the HTML
                const searchText = s.text.substring(0, 30);
                const idx = html.indexOf(searchText);
                if (idx !== -1) {
                    // Find the end of this text segment
                    const endIdx = idx + s.text.length;
                    const before = html.substring(0, idx);
                    const sentence = html.substring(idx, Math.min(endIdx, html.length));
                    const after = html.substring(Math.min(endIdx, html.length));

                    // Only wrap if not already in a tag
                    if (!before.endsWith('<') && !sentence.includes('<')) {
                        html = before + `<span class="sentence" data-sentence-id="${s.id}">${sentence}</span>` + after;
                    }
                }
            });

            this.chapterContent.innerHTML = html;

            // Fallback to deterministic rendering if HTML wrapping misses IDs
            // or reorders them (both break next/prev and sentence highlighting).
            if (!this.chapterSentenceAnchorsAreReliable(chapter)) {
                this.chapterContent.innerHTML = this.renderChapterSentenceFallback(chapter);
            }

            // Add click handlers to sentences
            this.chapterContent.querySelectorAll('.sentence').forEach(el => {
                el.addEventListener('click', () => {
                    const sentenceId = el.dataset.sentenceId;
                    const sentence = this.sentences.find(s => s.id === sentenceId);
                    if (sentence) {
                        // If autoplay is enabled, continue from clicked sentence.
                        this.playSentence(sentence, { singleShot: !this.autoplay });
                    }
                });
            });
        }

        // Update chapter list UI
        document.querySelectorAll('.chapter-item').forEach((el, i) => {
            el.classList.toggle('active', i === index);
        });

        // Prefetch audio for this chapter
        this.prefetchChapterAudio(chapter);
        this.scheduleVisiblePrefetch();
        this.scheduleProgressSync();
        this.closeSidebar();
    }

    async clearBook() {
        if (!confirm('Close this book?')) return;

        await this.syncProgressNow();
        this.stopPlayback();
        this.releaseWakeLock();

        this.currentBook = null;
        this.currentBookId = null;
        this.sentences = [];
        this.audioElements = {};
        this.audioStatus = {};
        this.currentChapter = 0;
        this.playbackQueue = [];

        this.welcomeScreen.style.display = 'flex';
        this.readerContainer.style.display = 'none';
        this.audioControls.style.display = 'none';
        this.clearBtn.style.display = 'none';
        this.fileInput.value = '';
        this.headerTitle.textContent = 'Library';
        if (this.authClient?.isAuthenticated()) {
            document.body.classList.add('home-shell-active');
        }
        if (this.authClient?.isAuthenticated()) {
            this.loadLibrary();
        }
    }

    // =========================================================================
    // Cloud Library
    // =========================================================================

    async uploadBookToCloud(file, book) {
        if (!this.authClient?.isAuthenticated() || !book?.book_id) return;

        const token = await this.authClient.getAccessToken();
        if (!token) return;

        this.coordinator.setAccessToken(token);

        let existing = null;
        try {
            const listResponse = await this.coordinator.getBooks();
            existing = (listResponse?.books || []).find(b => b.book_id === book.book_id) || null;
        } catch (error) {
            existing = null;
        }

        if (!existing) {
            const coverType = book.cover_data_url?.startsWith('data:image/png')
                ? 'image/png'
                : (book.cover_data_url?.startsWith('data:image/webp') ? 'image/webp' : 'image/jpeg');
            const uploadUrls = await this.coordinator.getBookUploadUrls(book.book_id, coverType);

            const epubPut = await fetch(uploadUrls.epub.upload_url, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/epub+zip' },
                body: file
            });
            if (!epubPut.ok) {
                throw new Error('Failed to upload EPUB to cloud storage');
            }

            let coverKey = null;
            let coverUrl = null;
            if (book.cover_data_url) {
                const coverBlob = this.dataUrlToBlob(book.cover_data_url);
                if (coverBlob) {
                    const coverPut = await fetch(uploadUrls.cover.upload_url, {
                        method: 'PUT',
                        headers: { 'Content-Type': coverBlob.type || uploadUrls.cover.content_type || 'image/jpeg' },
                        body: coverBlob
                    });
                    if (!coverPut.ok) {
                        throw new Error('Failed to upload cover to cloud storage');
                    }
                    coverKey = uploadUrls.cover.key;
                    coverUrl = uploadUrls.cover.final_url;
                }
            }

            await this.coordinator.saveBook({
                book_id: book.book_id,
                title: book.title,
                author: book.author || '',
                epub_key: uploadUrls.epub.key,
                epub_url: uploadUrls.epub.final_url,
                cover_key: coverKey,
                cover_url: coverUrl,
                total_chapters: book.chapters?.length || 0,
                total_sentences: book.sentences?.length || 0
            });
        } else {
            await this.coordinator.saveBook({
                book_id: book.book_id,
                title: book.title,
                author: book.author || '',
                epub_url: existing.epub_url,
                cover_url: existing.cover_url,
                total_chapters: book.chapters?.length || existing.total_chapters || 0,
                total_sentences: book.sentences?.length || existing.total_sentences || 0
            });
        }
    }

    async loadLibrary() {
        if (!this.authClient?.isAuthenticated()) return;

        try {
            const token = await this.authClient.getAccessToken();
            if (!token) return;
            this.coordinator.setAccessToken(token);

            const data = await this.coordinator.getBooks();
            this.libraryBooks = data?.books || [];
            this.renderLibrary(this.libraryBooks);
        } catch (error) {
            console.error('Failed to load cloud library:', error);
            this.renderLibrary(this.libraryBooks);
        }
    }

    renderLibrary(books) {
        if (!this.librarySection || !this.libraryGrid) return;

        const isLoggedIn = !!this.authClient?.isAuthenticated();
        const allBooks = Array.isArray(books) ? books : [];
        const query = this.librarySearchQuery;
        const filteredBooks = query
            ? allBooks.filter(book => {
                const title = (book.title || '').toLowerCase();
                const author = (book.author || '').toLowerCase();
                return title.includes(query) || author.includes(query);
            })
            : allBooks;
        const continueReadingBooks = allBooks
            .filter(book => {
                const p = book.reading_progress?.percentage || 0;
                return p > 0 && p < 100;
            })
            .slice(0, 3);

        if (this.welcomeScreen) {
            this.welcomeScreen.classList.toggle('library-mode', isLoggedIn);
        }
        if (this.libraryHomeShell) {
            this.libraryHomeShell.style.display = isLoggedIn ? 'grid' : 'none';
        }
        document.body.classList.toggle('home-shell-active', isLoggedIn && !this.currentBook);

        if (!isLoggedIn) {
            this.librarySection.style.display = 'none';
            this.libraryGrid.innerHTML = '';
            if (this.continueReadingSection) this.continueReadingSection.style.display = 'none';
            return;
        }

        this.librarySection.style.display = 'block';
        this.libraryGrid.innerHTML = '';

        if (this.continueReadingSection && this.continueReadingGrid) {
            if (continueReadingBooks.length > 0) {
                this.continueReadingSection.style.display = 'block';
                this.continueReadingGrid.innerHTML = '';
                continueReadingBooks.forEach(book => {
                    this.continueReadingGrid.appendChild(this.createBookCard(book, { compact: true, includeActions: false }));
                });
            } else {
                this.continueReadingSection.style.display = 'none';
                this.continueReadingGrid.innerHTML = '';
            }
        }

        if (filteredBooks.length === 0) {
            this.libraryGrid.innerHTML = '<div class="library-empty"><p>No books match your search.</p></div>';
            return;
        }

        filteredBooks.forEach(book => {
            this.libraryGrid.appendChild(this.createBookCard(book, { compact: false, includeActions: true }));
        });
    }

    createBookCard(book, options = {}) {
        const { compact = false, includeActions = true } = options;
        const card = document.createElement('div');
        card.className = `book-card ${compact ? 'continue-book-card' : 'home-library-card'}`;
        card.dataset.bookId = book.book_id;

        const readingProgress = book.reading_progress?.percentage || 0;
        const audioProgress = book.audio_progress?.percentage || 0;
        const audioLeftCount = book.audio_progress?.left_count ??
            Math.max(0, (book.total_sentences || 0) - (book.audio_progress?.ready_count || 0));
        const readingLeft = Math.max(0, 100 - readingProgress);
        const hue = this.hashStringToHue(book.title || book.book_id);
        const actionState = this.bookActionState.get(book.book_id) || {};
        const generating = !!actionState.generating;
        const downloading = !!actionState.downloading;
        const generateLabel = actionState.generateLabel || 'Generate Audio';
        const downloadLabel = actionState.downloadLabel || 'Load Audio Locally';
        const relativeUpdated = this.formatRelativeTime(
            book.reading_progress?.updated_at || book.updated_at || book.created_at || null
        );
        const audioBadge = audioProgress >= 100
            ? '<span class="home-badge home-badge-audio">Audio</span>'
            : (audioProgress > 0
                ? `<span class="home-badge home-badge-progress">${Math.round(audioProgress)}%</span>`
                : '<span class="home-badge home-badge-muted">No Audio</span>');

        if (compact) {
            card.innerHTML = `
                <div class="book-cover" style="background: linear-gradient(135deg, hsl(${hue}, 42%, 38%) 0%, hsl(${hue + 18}, 38%, 30%) 100%);">
                    ${book.cover_url
                        ? `<img src="${book.cover_url}" alt="${book.title}" onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
                           <div class="book-cover-placeholder" style="display:none;">
                               <span class="placeholder-title">${this.escapeHtml(book.title || 'Untitled')}</span>
                           </div>`
                        : `<div class="book-cover-placeholder">
                               <span class="placeholder-title">${this.escapeHtml((book.title || 'Untitled').split(' ').slice(0, 3).join(' '))}</span>
                           </div>`
                    }
                </div>
                <div class="book-info">
                    <div class="book-title-text">${this.escapeHtml(book.title || 'Untitled')}</div>
                    <div class="book-author">${this.escapeHtml(book.author || 'Unknown author')}</div>
                </div>
            `;
            card.addEventListener('click', () => this.openBookFromLibrary(book.book_id));
            return card;
        }

        card.innerHTML = `
            <div class="book-cover" style="background: linear-gradient(135deg, hsl(${hue}, 60%, 50%) 0%, hsl(${hue + 30}, 50%, 40%) 100%);">
                ${book.cover_url
                    ? `<img src="${book.cover_url}" alt="${book.title}" onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
                       <div class="book-cover-placeholder" style="display:none;">
                           <span class="placeholder-title">${this.escapeHtml(book.title || 'Untitled')}</span>
                       </div>`
                    : `<div class="book-cover-placeholder">
                           <span class="placeholder-title">${this.escapeHtml(book.title || 'Untitled')}</span>
                       </div>`
                }
            </div>
            <div class="book-info">
                <div class="book-title-text">${this.escapeHtml(book.title || 'Untitled')}</div>
                <div class="book-author">${this.escapeHtml(book.author || 'Unknown author')}</div>
                <div class="book-progress">
                    <div class="progress-row">
                        <span class="progress-label">Read</span>
                        <div class="progress-bar-wrapper">
                            <div class="progress-bar-fill reading" style="width: ${readingProgress}%"></div>
                        </div>
                        <span class="progress-text">${Math.round(readingProgress)}%</span>
                    </div>
                    <div class="progress-row">
                        <span class="progress-label">Audio</span>
                        <div class="progress-bar-wrapper">
                            <div class="progress-bar-fill audio" style="width: ${audioProgress}%"></div>
                        </div>
                        <span class="progress-text">${Math.round(audioProgress)}%</span>
                    </div>
                    <div class="progress-row">
                        <span class="progress-text">Left to read: ${Math.round(readingLeft)}%</span>
                    </div>
                    <div class="progress-row">
                        <span class="progress-text">Audio left: ${audioLeftCount} sentence${audioLeftCount === 1 ? '' : 's'}</span>
                    </div>
                </div>
                <div class="book-meta-row">
                    ${audioBadge}
                    <span class="home-meta-time">${this.escapeHtml(relativeUpdated || '')}</span>
                </div>
                <div class="book-actions" ${includeActions ? '' : 'style="display:none;"'}>
                    <button class="book-action-btn generate" data-action="generate" ${generating ? 'disabled' : ''}>
                        ${this.escapeHtml(generateLabel)}
                    </button>
                    <button class="book-action-btn download" data-action="download" ${(downloading || generating) ? 'disabled' : ''}>
                        ${this.escapeHtml(downloadLabel)}
                    </button>
                </div>
            </div>
        `;

        card.addEventListener('click', () => this.openBookFromLibrary(book.book_id));
        if (includeActions) {
            card.querySelector('[data-action="generate"]')?.addEventListener('click', (event) => {
                event.stopPropagation();
                this.generateAudioForBook(book.book_id);
            });
            card.querySelector('[data-action="download"]')?.addEventListener('click', (event) => {
                event.stopPropagation();
                this.downloadAudioForBook(book.book_id);
            });
        }
        return card;
    }

    formatRelativeTime(dateValue) {
        if (!dateValue) return '';
        const date = new Date(dateValue);
        if (Number.isNaN(date.getTime())) return '';

        const now = Date.now();
        const diffMs = now - date.getTime();
        if (diffMs < 60000) return 'Just now';
        if (diffMs < 3600000) return `${Math.floor(diffMs / 60000)}m ago`;
        if (diffMs < 86400000) return `${Math.floor(diffMs / 3600000)}h ago`;
        if (diffMs < 172800000) return 'Yesterday';
        if (diffMs < 604800000) return `${Math.floor(diffMs / 86400000)}d ago`;
        return date.toLocaleDateString();
    }

    hashStringToHue(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            hash = str.charCodeAt(i) + ((hash << 5) - hash);
        }
        return Math.abs(hash) % 360;
    }

    setBookActionState(bookId, patch = {}) {
        const current = this.bookActionState.get(bookId) || {};
        this.bookActionState.set(bookId, { ...current, ...patch });
        this.renderLibrary(this.libraryBooks);
    }

    isLikelyMobileClient() {
        return /Android|iPhone|iPad|iPod|Mobi/i.test(navigator.userAgent || '');
    }

    async checkLocalWorkerHealth(timeoutMs = 3500) {
        const workerUrl = this.config.localWorkerUrl || 'http://127.0.0.1:5001';
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), timeoutMs);
        try {
            const response = await fetch(`${workerUrl}/worker/health`, { signal: controller.signal });
            if (!response.ok) return false;
            const payload = await response.json().catch(() => null);
            return payload?.status === 'ok';
        } catch (error) {
            return false;
        } finally {
            clearTimeout(timeout);
        }
    }

    updateBookAudioProgressEstimate(bookId, readyCount, totalSentences) {
        const total = Math.max(0, Number(totalSentences || 0));
        const ready = Math.max(0, Math.min(Number(readyCount || 0), total || Number(readyCount || 0)));
        const percentage = total > 0 ? Math.min(100, (ready / total) * 100) : 0;
        const leftCount = Math.max(0, total - ready);

        const idx = this.libraryBooks.findIndex(b => b.book_id === bookId);
        if (idx === -1) return;

        const existing = this.libraryBooks[idx];
        this.libraryBooks[idx] = {
            ...existing,
            total_sentences: total || existing.total_sentences || 0,
            audio_progress: {
                ...(existing.audio_progress || {}),
                ready_count: ready,
                total_sentences: total || existing.total_sentences || 0,
                percentage,
                left_count: leftCount,
                left_percentage: Math.max(0, 100 - percentage)
            }
        };
        this.renderLibrary(this.libraryBooks);
    }

    async persistBookAudioProgress(bookId, readyCount, totalSentences) {
        try {
            if (!this.authClient?.isAuthenticated()) return;
            const token = await this.authClient.getAccessToken();
            if (!token) return;
            this.coordinator.setAccessToken(token);

            const audioPct = totalSentences > 0 ? Math.min(100, (readyCount / totalSentences) * 100) : 0;
            const response = await this.coordinator.updateBookProgress(bookId, {
                ready_audio_count: readyCount,
                total_sentences: totalSentences,
                audio_percentage: audioPct
            });
            if (response?.book) {
                const idx = this.libraryBooks.findIndex(b => b.book_id === response.book.book_id);
                if (idx >= 0) {
                    this.libraryBooks[idx] = response.book;
                } else {
                    this.libraryBooks.unshift(response.book);
                }
                this.renderLibrary(this.libraryBooks);
            }
        } catch (error) {
            console.log('Book audio progress sync skipped:', error?.message || error);
        }
    }

    async loadBookSentencesFromCloud(bookId) {
        const cached = this.bookSentenceCache.get(bookId);
        if (cached?.length) {
            return cached;
        }

        if (this.currentBookId === bookId && this.sentences?.length) {
            this.bookSentenceCache.set(bookId, this.sentences);
            return this.sentences;
        }

        const bookMetaResponse = await this.coordinator.getBook(bookId);
        const bookMeta = bookMetaResponse?.book;
        if (!bookMeta) {
            throw new Error('Book metadata not found');
        }

        const blob = await this.downloadCloudBookBlob(bookId, bookMeta);

        const parsed = await this.parseEPUB(blob, {
            bookId,
            sourceName: `${bookMeta.title || bookId}.epub`
        });

        this.bookSentenceCache.set(bookId, parsed.sentences || []);
        return parsed.sentences || [];
    }

    async generateAudioForBook(bookId) {
        if (!bookId) return;
        if (!this.authClient?.isAuthenticated()) {
            this.authClient?.login();
            return;
        }

        if (this.isLikelyMobileClient()) {
            alert('Audio generation is desktop-only. Run the local worker on your PC, then use this button there.');
            return;
        }

        const workerHealthy = await this.checkLocalWorkerHealth();
        if (!workerHealthy) {
            alert('Local worker is not reachable at 127.0.0.1:5001. Start Docker worker on your PC first.');
            return;
        }

        this.setBookActionState(bookId, {
            generating: true,
            generateLabel: 'Generating 0%',
            downloadLabel: 'Load Audio Locally'
        });

        try {
            const token = await this.authClient.getAccessToken();
            if (!token) throw new Error('Missing access token');
            this.coordinator.setAccessToken(token);

            const sentences = await this.loadBookSentencesFromCloud(bookId);
            if (!sentences.length) throw new Error('No sentences found for this book');
            const job = await this.startWorkerBookJob(bookId, sentences, token);
            const done = await this.waitForWorkerBookJob(bookId, job?.job_id, sentences.length);
            await this.persistBookAudioProgress(bookId, done.ready, done.total);
            await this.loadLibrary();
            alert(`Audio generation complete: ${done.ready}/${done.total} ready${done.failed ? `, ${done.failed} failed` : ''}.`);
        } catch (error) {
            console.error('Failed to generate book audio:', error);
            alert(`Failed to generate audio: ${error?.message || error}`);
        } finally {
            this.setBookActionState(bookId, {
                generating: false,
                generateLabel: 'Generate Audio',
                downloadLabel: 'Load Audio Locally'
            });
        }
    }

    async startWorkerBookJob(bookId, sentences, token) {
        const workerUrl = this.config.localWorkerUrl || 'http://127.0.0.1:5001';
        const payload = {
            book_id: bookId,
            upload_format: 'm4b',
            sentences: (sentences || []).map((sentence, index) => ({
                id: sentence.id || `s_${index}`,
                hash: this.coordinator.hashText(sentence.text || ''),
                text: sentence.text || '',
                sentence_index: Number.isFinite(sentence.sentence_index) ? sentence.sentence_index : index
            }))
        };

        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 45000);
        try {
            const response = await fetch(`${workerUrl}/worker/book/start`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify(payload),
                signal: controller.signal
            });
            const data = await response.json().catch(() => ({}));
            if (!response.ok || !data?.success || !data?.job?.job_id) {
                const err = data?.error || `Worker start failed (${response.status})`;
                throw new Error(err);
            }
            return data.job;
        } finally {
            clearTimeout(timeout);
        }
    }

    async fetchWorkerBookJob(jobId) {
        const workerUrl = this.config.localWorkerUrl || 'http://127.0.0.1:5001';
        const response = await fetch(`${workerUrl}/worker/book/status/${encodeURIComponent(jobId)}`);
        if (!response.ok) {
            throw new Error(`Failed to fetch worker job status (${response.status})`);
        }
        const data = await response.json();
        if (!data?.success || !data?.job) {
            throw new Error(data?.error || 'Invalid worker job status payload');
        }
        return data.job;
    }

    async waitForWorkerBookJob(bookId, jobId, fallbackTotal) {
        if (!jobId) throw new Error('Missing worker job id');

        let lastPersist = 0;
        let stable = {
            total: fallbackTotal || 0,
            ready: 0,
            failed: 0,
            status: 'queued'
        };

        const maxWaitMs = 6 * 60 * 60 * 1000;
        const startTime = Date.now();

        while (Date.now() - startTime < maxWaitMs) {
            const job = await this.fetchWorkerBookJob(jobId);
            const total = Math.max(0, Number(job.total || stable.total || 0));
            const ready = Math.max(0, Number(job.ready || 0));
            const failed = Math.max(0, Number(job.failed || 0));
            const processed = Math.max(0, Number(job.processed || (ready + failed)));
            const pct = total > 0 ? Math.min(100, Math.round((processed / total) * 100)) : 100;

            stable = {
                total,
                ready,
                failed,
                status: job.status || 'running'
            };

            this.setBookActionState(bookId, {
                generating: true,
                generateLabel: `Generating ${pct}%`,
                downloadLabel: 'Load Audio Locally'
            });
            this.updateBookAudioProgressEstimate(bookId, ready, total);

            const now = Date.now();
            if (now - lastPersist > 5000 || job.status === 'completed' || job.status === 'failed') {
                await this.persistBookAudioProgress(bookId, ready, total);
                lastPersist = now;
            }

            if (job.status === 'completed') {
                return stable;
            }
            if (job.status === 'failed') {
                const msg = job.last_error ? `Worker failed: ${job.last_error}` : 'Worker generation failed';
                throw new Error(msg);
            }

            await new Promise(resolve => setTimeout(resolve, 1200));
        }

        throw new Error('Worker generation timed out');
    }

    async downloadAudioForBook(bookId) {
        if (!bookId) return;
        if (!this.authClient?.isAuthenticated()) {
            this.authClient?.login();
            return;
        }

        this.setBookActionState(bookId, {
            downloading: true,
            generateLabel: 'Generate Audio',
            downloadLabel: 'Scanning 0%'
        });

        try {
            const token = await this.authClient.getAccessToken();
            if (!token) throw new Error('Missing access token');
            this.coordinator.setAccessToken(token);

            const sentences = await this.loadBookSentencesFromCloud(bookId);
            if (!sentences.length) throw new Error('No sentences found for this book');

            const chunkSize = 100;
            const readyItems = [];
            for (let i = 0; i < sentences.length; i += chunkSize) {
                const chunk = sentences.slice(i, i + chunkSize);
                const results = await this.coordinator.checkBatch(
                    chunk.map(s => ({
                        hash: this.coordinator.hashText(s.text || ''),
                        text: s.text || ''
                    }))
                );

                results.forEach((result, idx) => {
                    if (result?.status === 'ready' && result?.url) {
                        readyItems.push({
                            sentence: chunk[idx],
                            hash: result.hash,
                            url: result.url
                        });
                    }
                });

                const pct = Math.round((Math.min(i + chunkSize, sentences.length) / sentences.length) * 100);
                this.setBookActionState(bookId, {
                    downloading: true,
                    generateLabel: 'Generate Audio',
                    downloadLabel: `Scanning ${pct}%`
                });
            }

            if (!readyItems.length) {
                alert('No cloud audio is available yet for this book.');
                return;
            }

            for (let i = 0; i < readyItems.length; i++) {
                const item = readyItems[i];
                const proxyAudioUrl = `${this.coordinator.apiUrl}/audio/${encodeURIComponent(item.hash)}`;
                const response = await fetch(proxyAudioUrl);
                if (!response.ok) continue;
                const blob = await response.blob();
                if (item.sentence?.id) {
                    await this.setCachedAudioBlob(bookId, item.sentence.id, blob, 'sentence');
                }
                if (item.hash) {
                    await this.setCachedAudioBlob(bookId, item.hash, blob, 'hash');
                }

                if ((i + 1) % 20 === 0 || i + 1 === readyItems.length) {
                    const pct = Math.round(((i + 1) / readyItems.length) * 100);
                    this.setBookActionState(bookId, {
                        downloading: true,
                        generateLabel: 'Generate Audio',
                        downloadLabel: `Loading ${pct}%`
                    });
                }
            }

            alert(`Loaded ${readyItems.length} compressed audio files locally.`);
        } catch (error) {
            console.error('Failed to download book audio:', error);
            alert(`Failed to download audio files: ${error?.message || error}`);
        } finally {
            this.setBookActionState(bookId, {
                downloading: false,
                generateLabel: 'Generate Audio',
                downloadLabel: 'Load Audio Locally'
            });
        }
    }

    async openBookFromLibrary(bookId) {
        if (!bookId) return;
        if (!this.authClient?.isAuthenticated()) {
            this.authClient?.login();
            return;
        }

        this.showLoading('Opening book...');
        try {
            const token = await this.authClient.getAccessToken();
            if (!token) throw new Error('Missing access token');
            this.coordinator.setAccessToken(token);

            const bookMetaResponse = await this.coordinator.getBook(bookId);
            const bookMeta = bookMetaResponse?.book;
            if (!bookMeta) throw new Error('Book metadata not found');

            const blob = await this.downloadCloudBookBlob(bookId, bookMeta);
            const parsed = await this.parseEPUB(blob, {
                bookId,
                sourceName: `${bookMeta.title || bookId}.epub`
            });
            this.bookSentenceCache.set(bookId, parsed.sentences || []);

            this.currentBook = parsed;
            this.sentences = parsed.sentences;
            this.currentBookId = parsed.book_id;
            this.displayBook();
            this.applySavedProgress(bookMeta);
            this.prefetchAudio();
            this.scheduleProgressSync();
        } catch (error) {
            console.error('Error opening cloud book:', error);
            alert('Failed to open book from cloud library');
        } finally {
            this.hideLoading();
        }
    }

    async downloadCloudBookBlob(bookId, bookMeta) {
        const token = await this.authClient.getAccessToken();
        if (!token) throw new Error('Missing access token');

        // Prefer coordinator proxy endpoint on same API domain.
        const proxyUrl = `${this.coordinator.apiUrl}/books/${encodeURIComponent(bookId)}/file`;
        for (let attempt = 1; attempt <= 3; attempt++) {
            try {
                const blob = await this.fetchBlobWithTimeout(proxyUrl, {
                    method: 'GET',
                    headers: { 'Authorization': `Bearer ${token}` },
                    cache: 'no-store'
                }, 45000);
                if (blob) {
                    await this.setCachedBookBlob(bookId, blob, `${bookMeta?.title || bookId}.epub`);
                    return blob;
                }
            } catch (error) {
                // Retry with XHR path (helps when fetch is intercepted by extensions).
                try {
                    const xhrBlob = await this.fetchBlobViaXhr(proxyUrl, {
                        Authorization: `Bearer ${token}`
                    }, 45000);
                    if (xhrBlob) {
                        await this.setCachedBookBlob(bookId, xhrBlob, `${bookMeta?.title || bookId}.epub`);
                        return xhrBlob;
                    }
                } catch (xhrError) {
                    // Continue to next retry attempt.
                }
            }
            await new Promise(resolve => setTimeout(resolve, 300 * attempt));
        }

        const dl = await this.coordinator.getBookDownloadUrl(bookId);
        const candidateUrls = [
            bookMeta?.epub_url,
            dl?.public_url,
            dl?.download_url,
            dl?.signed_url
        ].filter(Boolean);

        for (const url of candidateUrls) {
            try {
                const response = await fetch(url, { cache: 'no-store' });
                if (response.ok) {
                    const blob = await response.blob();
                    await this.setCachedBookBlob(bookId, blob, `${bookMeta?.title || bookId}.epub`);
                    return blob;
                }
            } catch (error) {
                // Try next URL candidate.
            }
        }

        const localFallback = await this.getCachedBookBlob(bookId);
        if (localFallback) {
            console.warn('Using locally cached EPUB fallback for book', bookId);
            return localFallback;
        }

        throw new Error('Failed to download cloud EPUB (and no local cached copy available)');
    }

    async fetchBlobWithTimeout(url, options = {}, timeoutMs = 45000) {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), timeoutMs);
        try {
            const response = await fetch(url, { ...options, signal: controller.signal });
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            return await response.blob();
        } finally {
            clearTimeout(timeout);
        }
    }

    fetchBlobViaXhr(url, headers = {}, timeoutMs = 45000) {
        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            xhr.open('GET', url, true);
            xhr.responseType = 'blob';
            xhr.timeout = timeoutMs;

            Object.entries(headers || {}).forEach(([k, v]) => {
                try {
                    xhr.setRequestHeader(k, v);
                } catch (e) {
                    // Ignore invalid headers.
                }
            });

            xhr.onload = () => {
                if (xhr.status >= 200 && xhr.status < 300 && xhr.response) {
                    resolve(xhr.response);
                } else {
                    reject(new Error(`XHR ${xhr.status}`));
                }
            };
            xhr.onerror = () => reject(new Error('XHR network error'));
            xhr.ontimeout = () => reject(new Error('XHR timeout'));
            xhr.send();
        });
    }

    applySavedProgress(bookMeta = null) {
        const resolvedMeta = bookMeta || this.libraryBooks.find(b => b.book_id === this.currentBookId) || null;
        if (!resolvedMeta || !this.currentBook) return;

        const chapterIndex = Math.max(0, Math.min(
            this.currentBook.chapters.length - 1,
            Number(resolvedMeta.chapter_index || 0)
        ));

        this.displayChapter(chapterIndex);

        if (resolvedMeta.sentence_id) {
            const sentence = this.sentenceById.get(resolvedMeta.sentence_id);
            if (sentence) {
                this.currentSentence = sentence;
                setTimeout(() => {
                    const el = document.querySelector(`[data-sentence-id="${sentence.id}"]`);
                    if (el) {
                        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    }
                }, 100);
            }
        }
    }

    getCurrentAnchorSentence() {
        if (this.currentSentence) return this.currentSentence;
        const visible = this.getCurrentPageSentences(1);
        return visible[0] || null;
    }

    scheduleProgressSync() {
        if (!this.currentBookId || !this.authClient?.isAuthenticated()) return;
        if (this.progressSyncTimer) {
            clearTimeout(this.progressSyncTimer);
        }
        this.progressSyncTimer = setTimeout(() => this.syncProgressNow(), 1200);
    }

    async syncProgressNow() {
        if (!this.currentBookId || !this.currentBook || !this.authClient?.isAuthenticated()) return;

        try {
            const token = await this.authClient.getAccessToken();
            if (!token) return;
            this.coordinator.setAccessToken(token);

            const anchor = this.getCurrentAnchorSentence();
            const ready = Object.values(this.audioStatus).filter(s => s === 'ready').length;
            const totalSentences = this.sentences.length || 0;
            const totalChapters = this.currentBook.chapters?.length || 0;
            const readingPct = totalChapters > 0
                ? Math.min(100, ((this.currentChapter + 1) / totalChapters) * 100)
                : 0;
            const audioPct = totalSentences > 0 ? Math.min(100, (ready / totalSentences) * 100) : 0;

            const response = await this.coordinator.updateBookProgress(this.currentBookId, {
                chapter_index: this.currentChapter || 0,
                sentence_index: anchor?.sentence_index || 0,
                sentence_id: anchor?.id || null,
                total_chapters: totalChapters,
                total_sentences: totalSentences,
                reading_percentage: readingPct,
                ready_audio_count: ready,
                audio_percentage: audioPct
            });

            if (response?.book) {
                const idx = this.libraryBooks.findIndex(b => b.book_id === response.book.book_id);
                if (idx >= 0) {
                    this.libraryBooks[idx] = response.book;
                } else {
                    this.libraryBooks.unshift(response.book);
                }
                this.renderLibrary(this.libraryBooks);
            }
        } catch (error) {
            console.log('Progress sync skipped:', error?.message || error);
        }
    }

    // =========================================================================
    // Audio - Coordinator Integration
    // =========================================================================

    async prefetchAudio() {
        this.scheduleVisiblePrefetch();
    }

    async prefetchChapterAudio(chapter) {
        if (!chapter?.sentences?.length) return;
        this.scheduleVisiblePrefetch();
    }

    scheduleVisiblePrefetch() {
        if (this.prefetchTimer) {
            clearTimeout(this.prefetchTimer);
        }
        this.prefetchTimer = setTimeout(() => this.prefetchVisibleSentences(), 250);
    }

    getVisibleSentenceCandidates(limit = 10) {
        const elements = Array.from(document.querySelectorAll('#chapterContent .sentence[data-sentence-id]'));
        if (!elements.length) return [];

        const viewportHeight = window.innerHeight || document.documentElement.clientHeight || 800;
        const seen = new Set();
        const ranked = [];

        elements.forEach(el => {
            const sentenceId = el.dataset.sentenceId;
            if (!sentenceId || seen.has(sentenceId)) return;
            seen.add(sentenceId);

            const rect = el.getBoundingClientRect();
            const inView = rect.bottom >= 0 && rect.top <= viewportHeight;
            if (!inView) return;

            // Prioritize closer to reading focal point (upper-middle viewport).
            const distance = Math.abs(rect.top - viewportHeight * 0.35);
            ranked.push({ sentenceId, distance });
        });

        ranked.sort((a, b) => a.distance - b.distance);
        return ranked
            .slice(0, limit)
            .map(item => this.sentenceById.get(item.sentenceId))
            .filter(Boolean);
    }

    getCurrentChapterSentences() {
        const chapter = this.currentBook?.chapters?.[this.currentChapter];
        return chapter?.sentences || [];
    }

    getCurrentPageSentences(limit = 60) {
        const elements = Array.from(document.querySelectorAll('#chapterContent .sentence[data-sentence-id]'));
        if (!elements.length) return [];

        const viewportHeight = window.innerHeight || document.documentElement.clientHeight || 800;
        const seen = new Set();
        const ids = [];

        elements.forEach(el => {
            const rect = el.getBoundingClientRect();
            const inView = rect.bottom >= 0 && rect.top <= viewportHeight;
            if (!inView) return;
            const sentenceId = el.dataset.sentenceId;
            if (!sentenceId || seen.has(sentenceId)) return;
            seen.add(sentenceId);
            ids.push(sentenceId);
        });

        if (!ids.length) return [];

        const chapterSentences = this.getCurrentChapterSentences();
        const idSet = new Set(ids);
        return chapterSentences
            .filter(s => idSet.has(s.id))
            .slice(0, limit);
    }

    getActiveSentenceQueue() {
        return this.playbackQueue?.length ? this.playbackQueue : this.getCurrentChapterSentences();
    }

    refreshPlaybackQueue() {
        this.playbackQueue = this.getCurrentChapterSentences();
        return this.playbackQueue;
    }

    async prefetchVisibleSentences() {
        if (!this.currentBook || !this.sentences.length) return;

        let targets = this.getVisibleSentenceCandidates(10);
        if (!targets.length) {
            const chapter = this.currentBook.chapters?.[this.currentChapter];
            targets = (chapter?.sentences || []).slice(0, 6);
        }
        if (!targets.length) return;

        try {
            const results = await this.coordinator.prefetch(targets.map(s => ({ text: s.text })));
            results.forEach((result, i) => {
                const sentence = targets[i];
                if (!sentence) return;
                this.audioStatus[sentence.id] = result.status;
            });
            this.updateAudioProgress();

            // Background generation on local worker for current visible targets.
            if (this.authClient?.isAuthenticated()) {
                const token = await this.authClient.getAccessToken();
                if (token) {
                    for (let i = 0; i < targets.length; i++) {
                        const sentence = targets[i];
                        const result = results[i];
                        if (!sentence || !result) continue;
                        if (result.status === 'ready') continue;
                        if (this.generationPromises.size >= this.maxBackgroundGenerations) break;
                        this.generateAndUploadAudio(sentence, this.coordinator.hashText(sentence.text), token, true);
                    }
                }
            }
        } catch (error) {
            console.log('Visible prefetch failed:', error);
        }
    }

    async getAudioForSentence(sentence) {
        const hash = this.coordinator.hashText(sentence.text);
        const proxyAudioUrl = `${this.coordinator.apiUrl}/audio/${encodeURIComponent(hash)}`;

        // Check local cache first
        if (this.audioElements[sentence.id]) {
            return this.audioElements[sentence.id];
        }

        const bookId = this.currentBookId || '';
        let cacheId = sentence?.id ? this.makeAudioCacheId(bookId, sentence.id, 'sentence') : null;
        let cachedBlob = sentence?.id ? await this.getCachedAudioBlob(bookId, sentence.id, 'sentence') : null;
        if (!cachedBlob) {
            cacheId = this.makeAudioCacheId(bookId, hash, 'hash');
            cachedBlob = await this.getCachedAudioBlob(bookId, hash, 'hash');
        }
        if (cachedBlob) {
            const localUrl = this.getOrCreateBlobUrl(cacheId, cachedBlob);
            const audio = new Audio(localUrl);
            audio.playbackRate = this.playbackSpeed;
            this.audioElements[sentence.id] = audio;
            this.audioStatus[sentence.id] = 'ready';
            this.coordinator.audioCache.set(hash, localUrl);
            return audio;
        }

        let accessToken = null;
        // Try coordinator if authenticated
        if (this.authClient?.isAuthenticated()) {
            try {
                accessToken = await this.authClient.getAccessToken();
                this.coordinator.setAccessToken(accessToken);
                const result = await this.coordinator.checkAudio(sentence.text, hash);

                if (result.status === 'ready' && result.url) {
                    const audio = new Audio(result.url);
                    audio.addEventListener('error', () => {
                        if (audio.src !== proxyAudioUrl) {
                            audio.src = proxyAudioUrl;
                            audio.playbackRate = this.playbackSpeed;
                        }
                    }, { once: true });
                    audio.playbackRate = this.playbackSpeed;
                    this.audioElements[sentence.id] = audio;
                    this.audioStatus[sentence.id] = 'ready';
                    try {
                        const resp = await fetch(proxyAudioUrl);
                        if (resp.ok) {
                            const blob = await resp.blob();
                            await this.setCachedAudioBlob(bookId, hash, blob, 'hash');
                            if (sentence?.id) {
                                await this.setCachedAudioBlob(bookId, sentence.id, blob, 'sentence');
                            }
                        }
                    } catch (cacheError) {
                        console.log('Audio cache write skipped:', cacheError?.message || cacheError);
                    }
                    return audio;
                }

                // Not ready in coordinator: generate locally on worker and upload.
                const generatedUrl = await this.generateAndUploadAudio(sentence, hash, accessToken);
                if (generatedUrl) {
                    const audio = new Audio(generatedUrl);
                    audio.playbackRate = this.playbackSpeed;
                    this.audioElements[sentence.id] = audio;
                    this.audioStatus[sentence.id] = 'ready';
                    return audio;
                }
            } catch (error) {
                console.log('Coordinator not available, using browser TTS');
            }
        }

        // Fallback: Use browser speech synthesis
        return this.useBrowserTTS(sentence);
    }

    useBrowserTTS(sentence) {
        const reader = this;
        return new Promise((resolve) => {
            // Create a wrapper that properly handles events
            const ttsWrapper = {
                _endedHandlers: [],
                _playbackSpeed: this.playbackSpeed,

                play: function() {
                    return new Promise((playResolve) => {
                        // Cancel previous speech, but interruption is expected during transitions.
                        reader.cancelBrowserSpeech(true);

                        const utterance = new SpeechSynthesisUtterance(sentence.text);
                        utterance.rate = this._playbackSpeed;
                        utterance.lang = 'en-US';

                        utterance.onend = () => {
                            playResolve();
                            // Fire all registered ended handlers
                            this._endedHandlers.forEach(handler => handler());
                        };

                        utterance.onerror = (e) => {
                            const ignorable = e?.error === 'interrupted' ||
                                             e?.error === 'canceled' ||
                                             reader._expectSpeechInterruption;
                            if (!ignorable) {
                                console.warn('TTS error:', e);
                            }
                            playResolve();
                            // Do not advance autoplay for expected cancellation events.
                            if (!ignorable) {
                                this._endedHandlers.forEach(handler => handler());
                            }
                        };

                        speechSynthesis.speak(utterance);
                    });
                },

                pause: () => speechSynthesis.pause(),
                paused: false,
                currentTime: 0,

                get playbackRate() {
                    return this._playbackSpeed;
                },
                set playbackRate(val) {
                    this._playbackSpeed = val;
                },

                addEventListener: function(event, handler) {
                    if (event === 'ended') {
                        this._endedHandlers.push(handler);
                    }
                },

                removeEventListener: function(event, handler) {
                    if (event === 'ended') {
                        const idx = this._endedHandlers.indexOf(handler);
                        if (idx !== -1) this._endedHandlers.splice(idx, 1);
                    }
                }
            };

            this.audioElements[sentence.id] = ttsWrapper;
            this.audioStatus[sentence.id] = 'ready';
            resolve(ttsWrapper);
        });
    }

    async generateAndUploadAudio(sentence, hash, accessToken = null, background = false) {
        const workerUrl = this.config.localWorkerUrl || 'http://127.0.0.1:5001';
        if (!accessToken) {
            return null;
        }
        if (!sentence?.id) {
            return null;
        }

        const existingPromise = this.generationPromises.get(sentence.id);
        if (existingPromise) {
            return existingPromise;
        }

        if (background) {
            this.audioStatus[sentence.id] = 'generating';
            this.updateAudioProgress();
        }

        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 45000);

        const promise = (async () => {
            try {
                const response = await fetch(`${workerUrl}/worker/generate`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${accessToken}`
                    },
                    body: JSON.stringify({
                        hash,
                        text: sentence.text
                    }),
                    signal: controller.signal
                });

                if (!response.ok) {
                    return null;
                }

                const data = await response.json();
                if (data?.success && data?.url) {
                    this.coordinator.audioCache.set(hash, data.url);
                    this.audioStatus[sentence.id] = 'ready';
                    this.updateAudioProgress();
                    return data.url;
                }
            } catch (error) {
                // Local worker may not be running; silently fall back to browser TTS.
                console.log('Local worker unavailable for this sentence');
            } finally {
                clearTimeout(timeout);
                this.generationPromises.delete(sentence.id);
            }

            if (this.audioStatus[sentence.id] === 'generating') {
                this.audioStatus[sentence.id] = 'pending';
                this.updateAudioProgress();
            }
            return null;
        })();

        this.generationPromises.set(sentence.id, promise);
        return promise;
    }

    updateAudioProgress() {
        const ready = Object.values(this.audioStatus).filter(s => s === 'ready').length;
        const total = this.sentences.length;

        if (this.audioReady) this.audioReady.textContent = ready;
        if (this.audioTotal) this.audioTotal.textContent = total;
        if (this.audioProgressBar) {
            const pct = total > 0 ? (ready / total) * 100 : 0;
            this.audioProgressBar.style.width = `${pct}%`;
        }

        this.scheduleProgressSync();
    }

    // =========================================================================
    // Playback
    // =========================================================================

    stopCurrentAudio(reset = false) {
        if (this.currentAudio) {
            try {
                if (this.currentEndedHandler && this.currentAudio.removeEventListener) {
                    this.currentAudio.removeEventListener('ended', this.currentEndedHandler);
                }
            } catch (e) {
                // Ignore listener cleanup issues on wrapper audio objects.
            }

            try {
                this.currentAudio.pause();
                if (reset && this.currentAudio.currentTime !== undefined) {
                    this.currentAudio.currentTime = 0;
                }
            } catch (e) {
                // Ignore pause/reset errors for non-standard wrappers.
            }
        }

        this.currentAudio = null;
        this.currentEndedHandler = null;
    }

    stopAllAudio(reset = false) {
        Object.values(this.audioElements).forEach(audio => {
            try {
                audio.pause();
                if (reset && audio.currentTime !== undefined) {
                    audio.currentTime = 0;
                }
            } catch (e) {
                // Best effort stop.
            }
        });
    }

    async togglePlayPause() {
        if (this.isPlaying) {
            this.pausePlayback();
        } else {
            await this.startPlayback();
        }
    }

    async startPlayback() {
        if (!this.currentBook) return;

        this.isPlaying = true;
        this.lastPlayWasSingleShot = false;
        this.updatePlayPauseButton();
        this.requestWakeLock();

        // Snapshot a stable queue for the chapter; start from visible sentence.
        const queue = this.refreshPlaybackQueue();
        if (!this.currentSentence || !queue.find(s => s.id === this.currentSentence.id)) {
            const visible = this.getCurrentPageSentences(1);
            this.currentSentence = visible[0] || queue[0] || null;
        }

        if (this.currentSentence) {
            await this.playSentence(this.currentSentence, { singleShot: false });
        }
    }

    pausePlayback() {
        this.isPlaying = false;
        this.updatePlayPauseButton();
        this.stopCurrentAudio(false);
        this.stopAllAudio(false);
        this.scheduleProgressSync();
    }

    stopPlayback() {
        this.syncProgressNow();
        this.playbackRequestId += 1;  // Invalidate in-flight play requests
        this.isPlaying = false;
        this.currentSentence = null;
        this.playbackQueue = [];
        this.updatePlayPauseButton();
        this.releaseWakeLock();
        this.cancelBrowserSpeech(true);
        this.stopCurrentAudio(true);
        this.stopAllAudio(true);

        // Remove highlighting
        document.querySelectorAll('.sentence.playing').forEach(el => {
            el.classList.remove('playing');
        });
    }

    async playSentence(sentence, options = {}) {
        if (!sentence) return;
        const requestId = ++this.playbackRequestId;
        const singleShot = !!options.singleShot;
        this.lastPlayWasSingleShot = singleShot;
        if (singleShot) {
            // Clicking a sentence should only play this sentence.
            this.playbackQueue = [sentence];
        } else if (!this.playbackQueue.length) {
            this.refreshPlaybackQueue();
        }

        this.currentSentence = sentence;
        this.isPlaying = true;
        this.updatePlayPauseButton();
        this.scheduleProgressSync();

        // Stop any current speech synthesis
        this.cancelBrowserSpeech(true);
        this.stopCurrentAudio(true);
        this.stopAllAudio(true);

        // Highlight sentence
        document.querySelectorAll('.sentence.playing').forEach(el => {
            el.classList.remove('playing');
        });
        document.querySelectorAll(`[data-sentence-id="${sentence.id}"]`).forEach(el => {
            el.classList.add('playing');
            el.scrollIntoView({ behavior: 'smooth', block: 'center' });
        });

        // Get audio
        const audio = await this.getAudioForSentence(sentence);
        if (requestId !== this.playbackRequestId || this.currentSentence?.id !== sentence.id) {
            return;
        }

        if (audio) {
            // Prepare next sentence audio in background to reduce gaps.
            this.warmupNextSentenceAudio(sentence, singleShot);

            // Create a bound handler that we can remove later
            const endedHandler = () => {
                if (requestId !== this.playbackRequestId) {
                    return;
                }
                // Remove handler to avoid duplicates
                if (audio.removeEventListener) {
                    audio.removeEventListener('ended', endedHandler);
                }

                // Continue to next if still playing and autoplay is on
                if (this.isPlaying && this.autoplay && !singleShot) {
                    this.playNextSentence();
                }
            };

            // Set up ended handler
            if (audio.addEventListener) {
                audio.addEventListener('ended', endedHandler);
            }
            this.currentAudio = audio;
            this.currentEndedHandler = endedHandler;

            try {
                if (audio.playbackRate !== undefined) {
                    audio.playbackRate = this.playbackSpeed;
                }
                if (audio.currentTime !== undefined) {
                    audio.currentTime = 0;
                }
                await audio.play();
            } catch (error) {
                console.error('Audio playback failed:', error);
                if (requestId === this.playbackRequestId && this.isPlaying && this.autoplay && !singleShot) {
                    this.playNextSentence();
                }
            }
        } else if (this.isPlaying && this.autoplay && !singleShot) {
            // No audio available, skip to next
            setTimeout(() => this.playNextSentence(), 500);
        }
    }

    playNextSentence() {
        const queue = this.getActiveSentenceQueue();
        const chapterQueue = this.getCurrentChapterSentences();
        const effectiveQueue = queue.length > 1 ? queue : chapterQueue;
        const currentIndex = effectiveQueue.findIndex(s => s.id === this.currentSentence?.id);
        const nextIndex = currentIndex + 1;

        if (nextIndex >= 0 && nextIndex < effectiveQueue.length) {
            this.playSentence(effectiveQueue[nextIndex], { singleShot: false });
        } else {
            this.stopPlayback();
        }
    }

    playPrevSentence() {
        const queue = this.getActiveSentenceQueue();
        const chapterQueue = this.getCurrentChapterSentences();
        const effectiveQueue = queue.length > 1 ? queue : chapterQueue;
        const currentIndex = effectiveQueue.findIndex(s => s.id === this.currentSentence?.id);
        const prevIndex = currentIndex - 1;

        if (prevIndex >= 0) {
            this.playSentence(effectiveQueue[prevIndex], { singleShot: false });
        }
    }

    getNextSentenceForAutoplay(currentSentenceId) {
        const queue = this.getActiveSentenceQueue();
        const chapterQueue = this.getCurrentChapterSentences();
        const effectiveQueue = queue.length > 1 ? queue : chapterQueue;
        const currentIndex = effectiveQueue.findIndex(s => s.id === currentSentenceId);
        const nextIndex = currentIndex + 1;
        if (nextIndex < 0 || nextIndex >= effectiveQueue.length) return null;
        return effectiveQueue[nextIndex];
    }

    warmupNextSentenceAudio(currentSentence, singleShot = false) {
        if (!this.autoplay || singleShot || !currentSentence?.id) return;

        const nextSentence = this.getNextSentenceForAutoplay(currentSentence.id);
        if (!nextSentence || this.audioElements[nextSentence.id]) return;

        this.getAudioForSentence(nextSentence)
            .then((audio) => {
                if (!audio) return;
                // Prime network/decode path for HTMLAudio-backed sentences.
                if (audio.preload !== undefined) {
                    audio.preload = 'auto';
                }
                if (typeof audio.load === 'function') {
                    try {
                        audio.load();
                    } catch (e) {
                        // Best effort warmup.
                    }
                }
            })
            .catch(() => {
                // Ignore warmup failures; main playback path handles fallback.
            });
    }

    setPlaybackSpeed(value) {
        this.playbackSpeed = parseFloat(value);
        if (this.speedValue) this.speedValue.textContent = `${this.playbackSpeed.toFixed(1)}x`;

        // Update current audio
        Object.values(this.audioElements).forEach(audio => {
            audio.playbackRate = this.playbackSpeed;
        });

        localStorage.setItem('epub_playback_speed', this.playbackSpeed);
    }

    updatePlayPauseButton() {
        if (!this.playPauseBtn) return;

        const playIcon = this.playPauseBtn.querySelector('.icon-play');
        const pauseIcon = this.playPauseBtn.querySelector('.icon-pause');

        if (playIcon) playIcon.style.display = this.isPlaying ? 'none' : 'block';
        if (pauseIcon) pauseIcon.style.display = this.isPlaying ? 'block' : 'none';
    }

    // =========================================================================
    // Volunteer Mode
    // =========================================================================

    setVolunteerMode(enabled) {
        this.volunteerMode = enabled;
        localStorage.setItem('epub_volunteer_mode', enabled);

        if (enabled && this.authClient?.isAuthenticated()) {
            this.startVolunteerWorker();
        } else {
            this.stopVolunteerWorker();
        }
    }

    async startVolunteerWorker() {
        console.log('Starting volunteer worker...');
        // Would start background processing of audio generation tasks
    }

    stopVolunteerWorker() {
        console.log('Stopping volunteer worker');
    }

    // =========================================================================
    // UI Helpers
    // =========================================================================

    isMobileViewport() {
        return window.matchMedia('(max-width: 1024px)').matches;
    }

    openSidebar() {
        if (!this.leftSidebar) return;
        this.leftSidebar.classList.add('mobile-open');
        this.leftSidebar.classList.remove('hidden');
        this.sidebarBackdrop?.classList.add('visible');
        this.toggleChapters?.classList.add('active');
    }

    closeSidebar() {
        if (!this.leftSidebar) return;
        this.leftSidebar.classList.remove('mobile-open');
        this.sidebarBackdrop?.classList.remove('visible');
        this.toggleChapters?.classList.remove('active');
    }

    handleSidebarResponsive() {
        if (!this.isMobileViewport()) {
            this.closeSidebar();
            this.leftSidebar?.classList.remove('hidden');
        }
    }

    toggleSidebar() {
        if (!this.leftSidebar) return;

        if (this.isMobileViewport()) {
            const isOpen = this.leftSidebar.classList.contains('mobile-open');
            if (isOpen) {
                this.closeSidebar();
            } else {
                this.openSidebar();
            }
            return;
        }

        const hidden = this.leftSidebar.classList.toggle('hidden');
        this.toggleChapters?.classList.toggle('active', !hidden);
    }

    toggleSettingsPanel() {
        const isVisible = this.settingsPanel?.style.display !== 'none';
        if (this.settingsPanel) {
            this.settingsPanel.style.display = isVisible ? 'none' : 'block';
        }
    }

    toggleFocusMode() {
        this.focusModeActive = !this.focusModeActive;
        if (this.focusMode) {
            this.focusMode.style.display = this.focusModeActive ? 'flex' : 'none';
        }
    }

    setTheme(theme) {
        const existingClasses = document.body.className
            .split(/\s+/)
            .filter(Boolean)
            .filter(cls => !cls.startsWith('theme-'));

        if (theme !== 'light') {
            existingClasses.push(`theme-${theme}`);
        }

        document.body.className = existingClasses.join(' ');

        document.querySelectorAll('.theme-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.theme === theme);
        });

        localStorage.setItem('epub_theme', theme);
    }

    showLoading(text = 'Loading...') {
        if (this.loadingText) this.loadingText.textContent = text;
        if (this.loadingOverlay) this.loadingOverlay.style.display = 'flex';
    }

    hideLoading() {
        if (this.loadingOverlay) this.loadingOverlay.style.display = 'none';
    }

    loadSettings() {
        // Theme
        const savedTheme = localStorage.getItem('epub_theme') || 'light';
        this.setTheme(savedTheme);

        // Playback speed
        const savedSpeed = localStorage.getItem('epub_playback_speed');
        if (savedSpeed) {
            this.playbackSpeed = parseFloat(savedSpeed);
            if (this.speedSlider) this.speedSlider.value = this.playbackSpeed;
            if (this.speedValue) this.speedValue.textContent = `${this.playbackSpeed.toFixed(1)}x`;
        }

        // Autoplay - default to true for continuous reading
        const savedAutoplay = localStorage.getItem('epub_autoplay');
        this.autoplay = savedAutoplay === null ? true : savedAutoplay === 'true';
        if (this.autoplayCheckbox) this.autoplayCheckbox.checked = this.autoplay;

        // Volunteer mode
        const savedVolunteer = localStorage.getItem('epub_volunteer_mode') === 'true';
        if (this.volunteerToggle) this.volunteerToggle.checked = savedVolunteer;
        this.volunteerMode = savedVolunteer;
    }

    handleKeyboard(e) {
        // Don't handle if in input
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

        switch (e.code) {
            case 'Space':
                e.preventDefault();
                this.togglePlayPause();
                break;
            case 'ArrowRight':
                this.playNextSentence();
                break;
            case 'ArrowLeft':
                this.playPrevSentence();
                break;
            case 'Escape':
                if (this.focusModeActive) {
                    this.toggleFocusMode();
                }
                break;
        }
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.epubReader = new EPUBReaderCloud();
});
