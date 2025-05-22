'use client';

import dynamic from 'next/dynamic';

// Dynamically import the client-side webcam component
const VideoStreamer = dynamic(() => import('@/components/VideoStreamer'), {
  ssr: false,
});

export default function ClientVideoWrapper() {
  return <VideoStreamer />;
}
