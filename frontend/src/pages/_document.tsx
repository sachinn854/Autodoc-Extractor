import * as React from 'react';
import Document, {
  Html,
  Head,
  Main,
  NextScript,
  DocumentContext,
} from 'next/document';

export default function MyDocument() {
  return (
    <Html lang="en">
      <Head>
        {/* PWA primary color */}
        <meta name="theme-color" content="#3b82f6" />
        <link rel="shortcut icon" href="/favicon.ico" />
        
        {/* Google Fonts */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"
          rel="stylesheet"
        />
        
        {/* Meta tags */}
        <meta name="description" content="AI-Powered Restaurant Bill Processing - Extract menu items, prices, and insights from restaurant bills automatically" />
        <meta name="keywords" content="restaurant bill, receipt processing, OCR, expense tracking, AI, machine learning" />
        <meta name="author" content="Restaurant Bill Analyzer" />
        
        {/* Open Graph */}
        <meta property="og:type" content="website" />
        <meta property="og:title" content="Restaurant Bill Analyzer - AI Bill Processing" />
        <meta property="og:description" content="Upload restaurant bills to automatically extract menu items, prices, and get spending insights" />
        <meta property="og:site_name" content="Restaurant Bill Analyzer" />
        
        {/* Twitter */}
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:title" content="Restaurant Bill Analyzer" />
        <meta name="twitter:description" content="AI-Powered Restaurant Bill Processing & Insights" />
      </Head>
      <body>
        <Main />
        <NextScript />
      </body>
    </Html>
  );
}

// Simple getInitialProps without Material-UI/Emotion
MyDocument.getInitialProps = async (ctx: DocumentContext) => {
  const initialProps = await Document.getInitialProps(ctx);
  return initialProps;
};