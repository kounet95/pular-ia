import React from 'react';
import { AbsoluteFill, useCurrentFrame, useVideoConfig, interpolate, spring } from 'remotion';
import { BRAND } from '../brand';
import { Background } from './Background';
import { AnimatedTitle, AnimatedSubtitle } from './AnimatedTitle';

export const DonationCTA: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const heartScale = spring({ frame, fps, config: { damping: 8, mass: 0.5 } });
  const beat = 1 + Math.sin(frame / 6) * 0.05;

  const btnFrame = Math.max(0, frame - 40);
  const btnOpacity = interpolate(btnFrame, [0, 15], [0, 1], { extrapolateRight: 'clamp' });
  const btnScale = spring({ frame: btnFrame, fps, config: { damping: 11 } });
  const btnPulse = 1 + Math.sin(frame / 10) * 0.03;

  return (
    <AbsoluteFill>
      <Background variant="gold" />
      <AbsoluteFill
        style={{
          alignItems: 'center',
          justifyContent: 'center',
          padding: '0 80px',
          textAlign: 'center',
        }}
      >
        <div
          style={{
            fontSize: 110,
            transform: `scale(${heartScale * beat})`,
            marginBottom: 20,
          }}
        >
          💛
        </div>
        <AnimatedTitle delay={0} fontSize={56}>
          100% de ton don finance le projet
        </AnimatedTitle>
        <div style={{ height: 16 }} />
        <AnimatedSubtitle delay={14} fontSize={30}>
          Une IA qui comprend le pular, un patrimoine qui ne s'efface pas.
          Chaque contribution compte, chaque don compte.
        </AnimatedSubtitle>
        <div
          style={{
            marginTop: 46,
            opacity: btnOpacity,
            transform: `scale(${interpolate(btnScale, [0, 1], [0.7, 1]) * btnPulse})`,
            background: `linear-gradient(135deg, ${BRAND.gold}, ${BRAND.goldLight})`,
            color: BRAND.bgDark,
            fontFamily: BRAND.fontTitle,
            fontWeight: 800,
            fontSize: 34,
            padding: '20px 48px',
            borderRadius: 999,
            boxShadow: '0 12px 40px rgba(224,168,60,0.45)',
          }}
        >
          💛 Faire un don
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
