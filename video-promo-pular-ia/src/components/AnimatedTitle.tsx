import React from 'react';
import { useCurrentFrame, useVideoConfig, spring, interpolate } from 'remotion';
import { BRAND } from '../brand';

export const AnimatedTitle: React.FC<{
  children: React.ReactNode;
  delay?: number;
  fontSize?: number;
  color?: string;
  align?: 'left' | 'center';
}> = ({ children, delay = 0, fontSize = 64, color = BRAND.text, align = 'center' }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const local = Math.max(0, frame - delay);

  const scale = spring({ frame: local, fps, config: { damping: 14, mass: 0.6 } });
  const opacity = interpolate(local, [0, 12], [0, 1], { extrapolateRight: 'clamp' });
  const translateY = interpolate(scale, [0, 1], [30, 0]);

  return (
    <div
      style={{
        opacity,
        transform: `translateY(${translateY}px) scale(${interpolate(scale, [0, 1], [0.85, 1])})`,
        fontFamily: BRAND.fontTitle,
        fontWeight: 800,
        fontSize,
        color,
        textAlign: align,
        lineHeight: 1.15,
        textShadow: '0 4px 24px rgba(0,0,0,0.45)',
      }}
    >
      {children}
    </div>
  );
};

export const AnimatedSubtitle: React.FC<{
  children: React.ReactNode;
  delay?: number;
  fontSize?: number;
  color?: string;
  align?: 'left' | 'center';
}> = ({ children, delay = 0, fontSize = 32, color = BRAND.gris, align = 'center' }) => {
  const frame = useCurrentFrame();
  const local = Math.max(0, frame - delay);
  const opacity = interpolate(local, [0, 15], [0, 1], { extrapolateRight: 'clamp' });
  const translateY = interpolate(local, [0, 15], [16, 0], { extrapolateRight: 'clamp' });

  return (
    <div
      style={{
        opacity,
        transform: `translateY(${translateY}px)`,
        fontFamily: BRAND.fontBody,
        fontWeight: 500,
        fontSize,
        color,
        textAlign: align,
        lineHeight: 1.4,
      }}
    >
      {children}
    </div>
  );
};
