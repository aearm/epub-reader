/**
 * EPUB Reader Configuration
 * Auto-generated from Terraform outputs
 */

window.EPUB_READER_CONFIG = {
    // API Configuration
    apiUrl: 'https://api.reader.psybytes.com',
    localWorkerUrl: 'http://127.0.0.1:5001',

    // Cognito Configuration
    cognito: {
        userPoolId: 'eu-west-1_Vc73UMTPO',
        clientId: '4elcqdsla61mcpnagsrlqc0fn0',
        domain: 'https://epub-reader-fe777fa8.auth.eu-west-1.amazoncognito.com',
        region: 'eu-west-1'
    },

    // S3 Audio Bucket
    audioBucketUrl: 'https://epub-reader-audio-fe777fa8.s3.eu-west-1.amazonaws.com',

    // Feature Flags
    features: {
        volunteerMode: true,  // Allow users to generate audio
        offlineSupport: true  // Cache audio locally
    }
};
