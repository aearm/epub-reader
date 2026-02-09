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
        userPoolId: 'us-east-2_U0N41k1IT',
        clientId: '12c5fvb8b1q2dpunuc35vef7t4',
        domain: 'https://epub-reader-fe777fa8.auth.us-east-2.amazoncognito.com',
        region: 'us-east-2'
    },

    // S3 Audio Bucket
    audioBucketUrl: 'https://epub-reader-audio-useast2-fe777fa8.s3.us-east-2.amazonaws.com',

    // Feature Flags
    features: {
        volunteerMode: true,  // Allow users to generate audio
        offlineSupport: true  // Cache audio locally
    }
};
