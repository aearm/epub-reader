class EPUBReader {
    constructor() {
        this.currentBook = null;
        this.currentChapter = 0;
        this.currentSentence = null;
        this.audioQueue = [];
        this.isPlaying = false;
        this.autoPlay = false;
        this.playbackRate = 1.0;
        this.audioElements = {};
        this.sentences = [];
        this.currentAudio = null;

        this.initializeElements();
        this.attachEventListeners();
    }

    initializeElements() {
        // File upload
        this.fileInput = document.getElementById('fileInput');
        this.uploadBtn = document.getElementById('uploadBtn');
        this.clearBtn = document.getElementById('clearBtn');

        // Loading overlay
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.loadingText = document.getElementById('loadingText');
        this.progressFill = document.getElementById('progressFill');
        this.progressText = document.getElementById('progressText');

        // Main containers
        this.welcomeScreen = document.getElementById('welcomeScreen');
        this.readerContainer = document.getElementById('readerContainer');

        // Book content
        this.bookTitle = document.getElementById('bookTitle');
        this.chapterList = document.getElementById('chapterList');
        this.chapterContent = document.getElementById('chapterContent');

        // Audio controls
        this.audioControls = document.getElementById('audioControls');
        this.playPauseBtn = document.getElementById('playPauseBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.prevBtn = document.getElementById('prevBtn');
        this.nextBtn = document.getElementById('nextBtn');
        this.speedSlider = document.getElementById('speedSlider');
        this.speedValue = document.getElementById('speedValue');
        this.autoplayCheckbox = document.getElementById('autoplayCheckbox');
        this.currentSentenceText = document.getElementById('currentSentenceText');
    }

    attachEventListeners() {
        // File upload
        this.uploadBtn.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', (e) => this.handleFileUpload(e));
        this.clearBtn.addEventListener('click', () => this.clearBook());

        // Audio controls
        this.playPauseBtn.addEventListener('click', () => this.togglePlayPause());
        this.stopBtn.addEventListener('click', () => this.stopPlayback());
        this.prevBtn.addEventListener('click', () => this.playPreviousSentence());
        this.nextBtn.addEventListener('click', () => this.playNextSentence());
        this.speedSlider.addEventListener('input', (e) => this.updatePlaybackSpeed(e));
        this.autoplayCheckbox.addEventListener('change', (e) => this.toggleAutoplay(e));
    }

    async handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        // Show loading overlay
        this.showLoading('Uploading EPUB file...');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                this.updateLoadingText('Processing chapters and generating audio...');
                await this.loadBookContent();
            } else {
                alert('Error uploading file: ' + data.error);
                this.hideLoading();
            }
        } catch (error) {
            console.error('Upload error:', error);
            alert('Failed to upload file');
            this.hideLoading();
        }
    }

    async loadBookContent() {
        try {
            const response = await fetch('/book/content');
            const data = await response.json();

            this.currentBook = data;
            this.sentences = data.sentences;

            // Preload audio files
            this.updateLoadingText('Preloading audio files...');
            await this.preloadAudio();

            this.displayBook();
            this.hideLoading();
        } catch (error) {
            console.error('Error loading book:', error);
            alert('Failed to load book content');
            this.hideLoading();
        }
    }

    async preloadAudio() {
        const audioMapping = this.currentBook.audio_mapping;
        const bookId = this.currentBook.book_id;

        console.log('Book ID:', bookId);
        console.log('Audio mapping sample:', Object.keys(audioMapping).slice(0, 3));

        if (!bookId) {
            console.error('No book ID found');
            return;
        }

        let loaded = 0;
        const total = Object.keys(audioMapping).length;
        console.log(`Total audio files to load: ${total}`);

        for (const [sentenceId, audioInfo] of Object.entries(audioMapping)) {
            const audioUrl = `/audio/${bookId}/${sentenceId}`;
            const audio = new Audio(audioUrl);
            audio.preload = 'auto';
            audio.playbackRate = this.playbackRate;

            // Add event listeners
            audio.addEventListener('ended', () => this.onAudioEnded());
            audio.addEventListener('error', (e) => {
                console.error(`Audio error for ${sentenceId}:`, e);
                console.error('Audio URL was:', audioUrl);
            });

            audio.addEventListener('loadeddata', () => {
                console.log(`Audio loaded for ${sentenceId}`);
            });

            this.audioElements[sentenceId] = audio;

            loaded++;
            const progress = Math.round((loaded / total) * 100);
            this.progressFill.style.width = `${progress}%`;
            this.progressText.textContent = `Loading audio: ${loaded}/${total}`;
        }

        console.log('Audio elements created:', Object.keys(this.audioElements).length);
    }

    displayBook() {
        // Hide welcome screen, show reader
        this.welcomeScreen.style.display = 'none';
        this.readerContainer.style.display = 'flex';
        this.audioControls.style.display = 'block';
        this.clearBtn.style.display = 'inline-block';

        // Set book title
        this.bookTitle.textContent = this.currentBook.title;

        // Display chapters in sidebar
        this.displayChapterList();

        // Display first chapter
        if (this.currentBook.chapters.length > 0) {
            this.displayChapter(0);
        }
    }

    displayChapterList() {
        this.chapterList.innerHTML = '';

        this.currentBook.chapters.forEach((chapter, index) => {
            const li = document.createElement('li');
            li.textContent = chapter.title || `Chapter ${index + 1}`;
            li.dataset.chapterIndex = index;

            if (index === 0) li.classList.add('active');

            li.addEventListener('click', () => this.displayChapter(index));
            this.chapterList.appendChild(li);
        });
    }

    displayChapter(index) {
        this.currentChapter = index;
        const chapter = this.currentBook.chapters[index];

        // Update active chapter in sidebar
        document.querySelectorAll('.chapter-list li').forEach(li => {
            li.classList.remove('active');
        });
        document.querySelector(`[data-chapter-index="${index}"]`).classList.add('active');

        // Display chapter content
        this.chapterContent.innerHTML = chapter.html;

        // Add click handlers to sentences
        this.attachSentenceHandlers();
    }

    attachSentenceHandlers() {
        // The sentences are already wrapped in spans with data-sentence-id from the backend
        // Just attach click handlers
        const sentences = this.chapterContent.querySelectorAll('.sentence');
        console.log(`Found ${sentences.length} sentences in chapter`);

        sentences.forEach((element, index) => {
            element.addEventListener('click', (e) => {
                e.stopPropagation();
                const sentenceId = element.dataset.sentenceId;
                console.log('Clicked sentence ID:', sentenceId);
                this.playSentence(sentenceId);
            });
        });
    }

    async playSentence(sentenceId) {
        console.log('Playing sentence:', sentenceId);

        // Stop current playback
        if (this.currentAudio) {
            this.currentAudio.pause();
        }

        // Remove previous highlighting
        document.querySelectorAll('.sentence.playing').forEach(el => {
            el.classList.remove('playing');
        });

        // Highlight current sentence
        const sentenceElement = document.querySelector(`[data-sentence-id="${sentenceId}"]`);
        if (sentenceElement) {
            sentenceElement.classList.add('playing');
            sentenceElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }

        // Get sentence data
        const sentence = this.sentences.find(s => s.id === sentenceId);
        if (!sentence) {
            console.error('Sentence not found:', sentenceId);
            return;
        }

        this.currentSentence = sentence;
        this.currentSentenceText.textContent = sentence.text;

        // Play audio
        const audio = this.audioElements[sentenceId];
        console.log('Audio element:', audio);

        if (audio) {
            this.currentAudio = audio;
            audio.playbackRate = this.playbackRate;
            audio.currentTime = 0;

            try {
                const playPromise = audio.play();
                if (playPromise !== undefined) {
                    await playPromise;
                    console.log('Audio playing successfully');
                    this.isPlaying = true;
                    this.updatePlayPauseButton();
                }
            } catch (error) {
                console.error('Error playing audio:', error);
                // Try to load and play again
                audio.load();
                try {
                    await audio.play();
                    console.log('Audio playing after reload');
                    this.isPlaying = true;
                    this.updatePlayPauseButton();
                } catch (retryError) {
                    console.error('Retry failed:', retryError);
                    alert('Failed to play audio. Check browser console for details.');
                }
            }
        } else {
            console.error('No audio element found for sentence:', sentenceId);
        }
    }

    onAudioEnded() {
        // Remove highlighting
        document.querySelectorAll('.sentence.playing').forEach(el => {
            el.classList.remove('playing');
            el.classList.add('completed');
        });

        // If autoplay is enabled, play next sentence
        if (this.autoPlay) {
            this.playNextSentence();
        } else {
            this.isPlaying = false;
            this.updatePlayPauseButton();
        }
    }

    playNextSentence() {
        if (!this.currentSentence) {
            // Start from first sentence
            if (this.sentences.length > 0) {
                this.playSentence(this.sentences[0].id);
            }
            return;
        }

        const currentIndex = this.sentences.findIndex(s => s.id === this.currentSentence.id);
        if (currentIndex < this.sentences.length - 1) {
            this.playSentence(this.sentences[currentIndex + 1].id);
        } else {
            // End of book
            this.stopPlayback();
        }
    }

    playPreviousSentence() {
        if (!this.currentSentence) return;

        const currentIndex = this.sentences.findIndex(s => s.id === this.currentSentence.id);
        if (currentIndex > 0) {
            this.playSentence(this.sentences[currentIndex - 1].id);
        }
    }

    togglePlayPause() {
        if (this.currentAudio && this.isPlaying) {
            this.currentAudio.pause();
            this.isPlaying = false;
        } else if (this.currentAudio) {
            this.currentAudio.play();
            this.isPlaying = true;
        } else {
            // Start from beginning
            this.playNextSentence();
        }
        this.updatePlayPauseButton();
    }

    stopPlayback() {
        if (this.currentAudio) {
            this.currentAudio.pause();
            this.currentAudio.currentTime = 0;
        }

        this.isPlaying = false;
        this.currentSentence = null;
        this.currentSentenceText.textContent = '';

        // Remove all highlighting
        document.querySelectorAll('.sentence.playing, .sentence.completed').forEach(el => {
            el.classList.remove('playing', 'completed');
        });

        this.updatePlayPauseButton();
    }

    updatePlaybackSpeed(event) {
        this.playbackRate = parseFloat(event.target.value);
        this.speedValue.textContent = `${this.playbackRate}x`;

        // Update all audio elements
        Object.values(this.audioElements).forEach(audio => {
            audio.playbackRate = this.playbackRate;
        });
    }

    toggleAutoplay(event) {
        this.autoPlay = event.target.checked;

        // If turning on autoplay and not currently playing, start
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

    async clearBook() {
        if (confirm('Are you sure you want to clear the current book?')) {
            // Stop playback
            this.stopPlayback();

            // Clear data
            this.currentBook = null;
            this.sentences = [];
            this.audioElements = {};

            // Reset UI
            this.welcomeScreen.style.display = 'flex';
            this.readerContainer.style.display = 'none';
            this.audioControls.style.display = 'none';
            this.clearBtn.style.display = 'none';
            this.fileInput.value = '';

            // Clear server
            await fetch('/clear');
        }
    }

    showLoading(text) {
        this.loadingOverlay.style.display = 'flex';
        this.loadingText.textContent = text;
        this.progressFill.style.width = '0%';
        this.progressText.textContent = '';
    }

    updateLoadingText(text) {
        this.loadingText.textContent = text;
    }

    hideLoading() {
        this.loadingOverlay.style.display = 'none';
    }
}

// Initialize the app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new EPUBReader();
});