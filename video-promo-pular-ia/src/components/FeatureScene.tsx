import React from 'react';
import { AbsoluteFill, useCurrentFrame, useVideoConfig, spring, interpolate } from 'remotion';
import { BRAND } from '../brand';
import { Background } from './Background';
import { AnimatedTitle, AnimatedSubtitle } from './AnimatedTitle';

export const FeatureScene: React.FC<{
  emoji: string;
  title: string;
  description: string;
  badge?: string;
  accent?: 'gold' | 'magenta';
}> = ({ emoji, title, description, badge, accent = 'gold' }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const iconScale = spring({ frame, fps, config: { damping: 10, mass: 0.7 } });
  const iconOpacity = interpolate(frame, [0, 10], [0, 1], { extrapolateRight: 'clamp' });
  const accentColor = accent === 'gold' ? BRAND.gold : BRAND.magentaLight;

  return (
    <AbsoluteFill>
      <Background variant={accent} />
      <AbsoluteFill
        style={{
          alignItems: 'center',
          justifyContent: 'center',
          padding: '0 90px',
        }}
      >
        {badge && (
          <div
            style={{
              opacity: interpolate(frame, [0, 10], [0, 1], { extrapolateRight: 'clamp' }),
              fontFamily: BRAND.fontBody,
              fontWeight: 700,
              fontSize: 24,
              letterSpacing: 2,
              color: accentColor,
              textTransform: 'uppercase',
              marginBottom: 24,
              border: `2px solid ${accentColor}`,
              borderRadius: 999,
              padding: '8px 24px',
            }}
          >
            {badge}
          </div>
        )}
        <div
          style={{
            fontSize: 140,
            opacity: iconOpacity,
            transform: `scale(${iconScale})`,
            marginBottom: 30,
            filter: 'drop-shadow(0 10px 30px rgba(0,0,0,0.4))',
          }}
        >
          {emoji}
        </div>
        <AnimatedTitle delay={6} fontSize={62}>
          {title}
        </AnimatedTitle>
        <div style={{ height: 22 }} />
        <AnimatedSubtitle delay={14} fontSize={30}>
          {description}
        </AnimatedSubtitle>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
