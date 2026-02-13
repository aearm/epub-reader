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
        userPoolId: 'eu-west-1_09eDKsYyu',
        clientId: '1tgsjl3qo9cbb0gvonkhfvf31n',
        domain: 'https://epub-reader-40620d6b.auth.eu-west-1.amazoncognito.com',
        region: 'eu-west-1'
    },

    // S3 Audio Bucket
    audioBucketUrl: 'https://epub-reader-audio-40620d6b.s3.eu-west-1.amazonaws.com',

    // Feature Flags
    features: {
        volunteerMode: true,  // Allow users to generate audio
        offlineSupport: true  // Cache audio locally
    }
};
