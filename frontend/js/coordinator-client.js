/**
 * Coordinator Client
 *
 * Handles communication with the distributed audio coordinator:
 * - Check if audio exists
 * - Get audio URLs from S3
 * - Upload generated audio (volunteer mode)
 */

class CoordinatorClient {
    constructor(config = {}) {
        this.apiUrl = config.apiUrl || 'https://api.reader.psybytes.com';
        this.audioCache = new Map();  // Local cache: hash -> url
        this.pendingRequests = new Map();  // Dedupe in-flight requests
        this.accessToken = null;
        this.onAuthRequired = config.onAuthRequired || (() => {});
    }

    /**
     * Set the access token for API calls
     */
    setAccessToken(token) {
        this.accessToken = token;
    }

    /**
     * Make authenticated API request
     */
    async apiRequest(endpoint, options = {}) {
        if (!this.accessToken) {
            this.onAuthRequired();
            throw new Error('Not authenticated');
        }
        const controller = new AbortController();
        const timeoutMs = options.timeoutMs || 15000;
        const timeout = setTimeout(() => controller.abort(), timeoutMs);

        let response;
        try {
            response = await fetch(`${this.apiUrl}${endpoint}`, {
                ...options,
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.accessToken}`,
                    ...options.headers
                },
                signal: controller.signal
            });
        } catch (error) {
            const msg = error?.name === 'AbortError'
                ? `API request timeout for ${endpoint}`
                : `API network error for ${endpoint}`;
            throw new Error(msg);
        } finally {
            clearTimeout(timeout);
        }

        if (response.status === 401) {
            this.onAuthRequired();
            throw new Error('Authentication expired');
        }

        if (!response.ok) {
            const body = await response.text().catch(() => '');
            throw new Error(`API ${endpoint} failed (${response.status}) ${body}`.trim());
        }

        return response.json();
    }

    /**
     * Generate hash for sentence text
     */
    hashText(text) {
        // Simple hash using Web Crypto API
        const encoder = new TextEncoder();
        const data = encoder.encode(text.trim());

        // Use a simple hash for now (replace with SHA-256 in production)
        let hash = 0;
        for (let i = 0; i < data.length; i++) {
            const char = data[i];
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash;
        }

        return Math.abs(hash).toString(16).padStart(16, '0').slice(0, 16);
    }

    /**
     * Check if audio exists for a sentence
     * Returns: { status: 'ready'|'pending'|'scheduled', url: string|null }
     */
    async checkAudio(text, hash = null) {
        const sentenceHash = hash || this.hashText(text);

        // Check local cache first
        if (this.audioCache.has(sentenceHash)) {
            return {
                hash: sentenceHash,
                status: 'ready',
                url: this.audioCache.get(sentenceHash)
            };
        }

        // Dedupe in-flight requests
        if (this.pendingRequests.has(sentenceHash)) {
            return this.pendingRequests.get(sentenceHash);
        }

        // Make API request
        const requestPromise = this.apiRequest('/check', {
            method: 'POST',
            body: JSON.stringify({ hash: sentenceHash, text })
        }).then(result => {
            this.pendingRequests.delete(sentenceHash);

            if (result.status === 'ready' && result.url) {
                this.audioCache.set(sentenceHash, result.url);
            }

            return result;
        }).catch(err => {
            this.pendingRequests.delete(sentenceHash);
            throw err;
        });

        this.pendingRequests.set(sentenceHash, requestPromise);
        return requestPromise;
    }

    /**
     * Check multiple sentences at once (batch)
     */
    async checkBatch(sentences) {
        // Filter out cached ones
        const uncached = sentences.filter(s => {
            const hash = s.hash || this.hashText(s.text);
            return !this.audioCache.has(hash);
        });

        if (uncached.length === 0) {
            // All cached
            return sentences.map(s => {
                const hash = s.hash || this.hashText(s.text);
                return {
                    hash,
                    status: 'ready',
                    url: this.audioCache.get(hash)
                };
            });
        }

        // Check with coordinator
        const result = await this.apiRequest('/check_batch', {
            method: 'POST',
            body: JSON.stringify({
                sentences: uncached.map(s => ({
                    hash: s.hash || this.hashText(s.text),
                    text: s.text
                }))
            })
        });

        // Update cache
        for (const item of result.results) {
            if (item.status === 'ready' && item.url) {
                this.audioCache.set(item.hash, item.url);
            }
        }

        // Return combined results
        return sentences.map(s => {
            const hash = s.hash || this.hashText(s.text);

            if (this.audioCache.has(hash)) {
                return { hash, status: 'ready', url: this.audioCache.get(hash) };
            }

            const found = result.results.find(r => r.hash === hash);
            return found || { hash, status: 'scheduled', url: null };
        });
    }

    /**
     * Prefetch audio for upcoming sentences
     */
    async prefetch(sentences) {
        const results = await this.checkBatch(sentences);

        // Preload ready audio
        for (const result of results) {
            if (result.status === 'ready' && result.url) {
                // Preload audio file
                const audio = new Audio();
                audio.preload = 'auto';
                audio.src = result.url;
            }
        }

        return results;
    }

    /**
     * Get audio URL, waiting if necessary
     */
    async getAudioUrl(text, maxWaitMs = 30000) {
        const hash = this.hashText(text);
        const startTime = Date.now();

        while (Date.now() - startTime < maxWaitMs) {
            const result = await this.checkAudio(text, hash);

            if (result.status === 'ready' && result.url) {
                return result.url;
            }

            // Wait before retry
            await new Promise(resolve => setTimeout(resolve, 1000));
        }

        return null;  // Timeout
    }

    // =========================================================================
    // Volunteer Mode (for desktop clients generating audio)
    // =========================================================================

    /**
     * Get tasks to generate audio for
     */
    async getTasks(limit = 10) {
        return this.apiRequest(`/tasks?limit=${limit}`);
    }

    /**
     * Get presigned URL for uploading audio to S3
     */
    async getUploadUrl(hash, format = 'wav') {
        return this.apiRequest('/upload_url', {
            method: 'POST',
            body: JSON.stringify({ hash, format })
        });
    }

    /**
     * Upload audio file to S3
     */
    async uploadAudio(hash, audioBlob, format = 'wav') {
        // Get presigned URL
        const { upload_url, final_url, content_type } = await this.getUploadUrl(hash, format);

        // Upload to S3
        await fetch(upload_url, {
            method: 'PUT',
            body: audioBlob,
            headers: {
                'Content-Type': content_type || (format === 'mp3' ? 'audio/mpeg' : 'audio/wav')
            }
        });

        // Mark task as complete
        await this.apiRequest('/complete', {
            method: 'POST',
            body: JSON.stringify({ hash, s3_url: final_url })
        });

        // Update local cache
        this.audioCache.set(hash, final_url);

        return final_url;
    }

    // =========================================================================
    // User Cloud Library
    // =========================================================================

    async getBooks() {
        return this.apiRequest('/books', { method: 'GET' });
    }

    async getBook(bookId) {
        return this.apiRequest(`/books/${encodeURIComponent(bookId)}`, { method: 'GET' });
    }

    async getBookUploadUrls(bookId, coverContentType = 'image/jpeg') {
        return this.apiRequest('/books/upload_urls', {
            method: 'POST',
            body: JSON.stringify({
                book_id: bookId,
                cover_content_type: coverContentType
            })
        });
    }

    async saveBook(metadata) {
        return this.apiRequest('/books', {
            method: 'POST',
            body: JSON.stringify(metadata || {})
        });
    }

    async getBookDownloadUrl(bookId) {
        return this.apiRequest(`/books/${encodeURIComponent(bookId)}/download_url`, {
            method: 'GET'
        });
    }

    async updateBookProgress(bookId, progress) {
        return this.apiRequest(`/books/${encodeURIComponent(bookId)}/progress`, {
            method: 'POST',
            body: JSON.stringify(progress || {})
        });
    }

    async deleteBook(bookId) {
        return this.apiRequest(`/books/${encodeURIComponent(bookId)}`, {
            method: 'DELETE'
        });
    }

    async generateBookAudio(bookId, sentences) {
        return this.apiRequest(`/books/${encodeURIComponent(bookId)}/generate_audio`, {
            method: 'POST',
            body: JSON.stringify({
                sentences: Array.isArray(sentences) ? sentences : []
            })
        });
    }

    async getBookAudioProgress(bookId) {
        return this.apiRequest(`/books/${encodeURIComponent(bookId)}/audio_progress`, {
            method: 'GET'
        });
    }

    async getBookAudioManifest(bookId, limit = 10000) {
        return this.apiRequest(`/books/${encodeURIComponent(bookId)}/audio_manifest?limit=${limit}`, {
            method: 'GET'
        });
    }

    /**
     * Get coordinator statistics
     */
    async getStats() {
        const response = await fetch(`${this.apiUrl}/stats`);
        return response.json();
    }
}

// Export for use in main app
window.CoordinatorClient = CoordinatorClient;
