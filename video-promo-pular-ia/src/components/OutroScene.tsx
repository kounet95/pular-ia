import React from 'react';
import { AbsoluteFill, useCurrentFrame, interpolate } from 'remotion';
import { BRAND } from '../brand';
import { Background } from './Background';
import { AnimatedTitle, AnimatedSubtitle } from './AnimatedTitle';

export const OutroScene: React.FC = () => {
  const frame = useCurrentFrame();

  const step2Frame = Math.max(0, frame - 40);
  const step2Opacity = interpolate(step2Frame, [0, 15], [0, 1], { extrapolateRight: 'clamp' });

  const linkFrame = Math.max(0, frame - 70);
  const linkOpacity = interpolate(linkFrame, [0, 15], [0, 1], { extrapolateRight: 'clamp' });

  return (
    <AbsoluteFill>
      <Background />
      <AbsoluteFill
        style={{
          alignItems: 'center',
          justifyContent: 'center',
          padding: '0 70px',
          textAlign: 'center',
        }}
      >
        <AnimatedTitle delay={0} fontSize={50}>
          1️⃣ Contribue ta voix
        </AnimatedTitle>
        <div
          style={{
            opacity: step2Opacity,
            fontFamily: BRAND.fontTitle,
            fontWeight: 800,
            fontSize: 50,
            color: BRAND.text,
            textAlign: 'center',
            marginTop: 10,
          }}
        >
          2️⃣ Fais un don ou partage
        </div>
        <div style={{ height: 34 }} />
        <AnimatedSubtitle delay={30} fontSize={28}>
          Pular IA — Corpus communautaire, open-source, porté par toi.
        </AnimatedSubtitle>
        <div
          style={{
            opacity: linkOpacity,
            marginTop: 30,
            fontFamily: BRAND.fontBody,
            fontWeight: 700,
            fontSize: 30,
            color: BRAND.gold,
            letterSpacing: 1,
          }}
        >
          {BRAND.website}
        </div>
        <div
          style={{
            opacity: linkOpacity,
            marginTop: 16,
            fontFamily: BRAND.fontBody,
            fontWeight: 600,
            fontSize: 24,
            color: BRAND.magentaLight,
          }}
        >
          Baŋ-baŋ! 🙏
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
