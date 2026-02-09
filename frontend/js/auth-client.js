/**
 * Cognito Authentication Client
 *
 * Handles user authentication with AWS Cognito:
 * - Login with hosted UI
 * - Token management
 * - Session persistence
 */

class AuthClient {
    constructor(config) {
        this.userPoolId = config.userPoolId;
        this.clientId = config.clientId;
        this.region = config.region || 'us-east-2';
        this.domain = config.domain;
        this.redirectUri = config.redirectUri || window.location.origin;

        this.accessToken = null;
        this.idToken = null;
        this.refreshToken = null;
        this.user = null;

        this.onAuthChange = config.onAuthChange || (() => {});

        // Try to restore session from storage
        this.restoreSession();
    }

    /**
     * Get Cognito hosted UI URL
     */
    getLoginUrl() {
        const params = new URLSearchParams({
            client_id: this.clientId,
            response_type: 'code',
            scope: 'email openid profile',
            redirect_uri: this.redirectUri
        });

        return `${this.domain}/login?${params.toString()}`;
    }

    /**
     * Get logout URL
     */
    getLogoutUrl() {
        const params = new URLSearchParams({
            client_id: this.clientId,
            logout_uri: this.redirectUri
        });

        return `${this.domain}/logout?${params.toString()}`;
    }

    /**
     * Redirect to login
     */
    login() {
        window.location.href = this.getLoginUrl();
    }

    /**
     * Logout user
     */
    logout() {
        this.clearSession();
        window.location.href = this.getLogoutUrl();
    }

    /**
     * Handle OAuth callback (exchange code for tokens)
     */
    async handleCallback() {
        const urlParams = new URLSearchParams(window.location.search);
        const code = urlParams.get('code');

        if (!code) {
            return false;
        }

        try {
            // Exchange code for tokens
            const response = await fetch(`${this.domain}/oauth2/token`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded'
                },
                body: new URLSearchParams({
                    grant_type: 'authorization_code',
                    client_id: this.clientId,
                    code: code,
                    redirect_uri: this.redirectUri
                })
            });

            if (!response.ok) {
                throw new Error('Token exchange failed');
            }

            const tokens = await response.json();
            this.setTokens(tokens);

            // Clean URL
            window.history.replaceState({}, document.title, window.location.pathname);

            return true;
        } catch (error) {
            console.error('Auth callback error:', error);
            return false;
        }
    }

    /**
     * Set tokens and decode user info
     */
    setTokens(tokens) {
        this.accessToken = tokens.access_token;
        this.idToken = tokens.id_token;
        this.refreshToken = tokens.refresh_token;

        // Decode user info from ID token
        if (this.idToken) {
            try {
                const payload = JSON.parse(atob(this.idToken.split('.')[1]));
                this.user = {
                    sub: payload.sub,
                    email: payload.email,
                    name: payload.name || payload.email
                };
            } catch (e) {
                console.error('Error decoding ID token:', e);
            }
        }

        // Save to storage
        this.saveSession();

        // Notify listeners
        this.onAuthChange(this.user);
    }

    /**
     * Refresh access token
     */
    async refreshAccessToken() {
        if (!this.refreshToken) {
            return false;
        }

        try {
            const response = await fetch(`${this.domain}/oauth2/token`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded'
                },
                body: new URLSearchParams({
                    grant_type: 'refresh_token',
                    client_id: this.clientId,
                    refresh_token: this.refreshToken
                })
            });

            if (!response.ok) {
                throw new Error('Token refresh failed');
            }

            const tokens = await response.json();
            this.setTokens({
                ...tokens,
                refresh_token: this.refreshToken  // Keep existing refresh token
            });

            return true;
        } catch (error) {
            console.error('Token refresh error:', error);
            this.clearSession();
            return false;
        }
    }

    /**
     * Get current access token (refresh if needed)
     */
    async getAccessToken() {
        if (!this.accessToken) {
            return null;
        }

        // Check if token is expired
        try {
            const payload = JSON.parse(atob(this.accessToken.split('.')[1]));
            const exp = payload.exp * 1000;

            if (Date.now() > exp - 60000) {  // Refresh 1 minute before expiry
                const refreshed = await this.refreshAccessToken();
                if (!refreshed) {
                    return null;
                }
            }
        } catch (e) {
            console.error('Error checking token expiry:', e);
        }

        return this.accessToken;
    }

    /**
     * Check if user is authenticated
     */
    isAuthenticated() {
        return !!this.accessToken;
    }

    /**
     * Get current user
     */
    getUser() {
        return this.user;
    }

    /**
     * Save session to localStorage
     */
    saveSession() {
        const session = {
            accessToken: this.accessToken,
            idToken: this.idToken,
            refreshToken: this.refreshToken,
            user: this.user
        };
        localStorage.setItem('epub_reader_auth', JSON.stringify(session));
    }

    /**
     * Restore session from localStorage
     */
    restoreSession() {
        try {
            const stored = localStorage.getItem('epub_reader_auth');
            if (stored) {
                const session = JSON.parse(stored);
                this.accessToken = session.accessToken;
                this.idToken = session.idToken;
                this.refreshToken = session.refreshToken;
                this.user = session.user;

                // Notify listeners
                if (this.user) {
                    this.onAuthChange(this.user);
                }
            }
        } catch (e) {
            console.error('Error restoring session:', e);
        }
    }

    /**
     * Clear session
     */
    clearSession() {
        this.accessToken = null;
        this.idToken = null;
        this.refreshToken = null;
        this.user = null;
        localStorage.removeItem('epub_reader_auth');
        this.onAuthChange(null);
    }
}

// Export for use in main app
window.AuthClient = AuthClient;
