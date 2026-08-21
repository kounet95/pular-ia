import React from 'react';
import { AbsoluteFill, useCurrentFrame, useVideoConfig, spring, interpolate } from 'remotion';
import { BRAND } from '../brand';
import { Background } from './Background';
import { AnimatedSubtitle } from './AnimatedTitle';

export const IntroScene: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const moonScale = spring({ frame, fps, config: { damping: 9, mass: 0.8 } });
  const moonOpacity = interpolate(frame, [0, 12], [0, 1], { extrapolateRight: 'clamp' });

  const logoFrame = Math.max(0, frame - 14);
  const logoOpacity = interpolate(logoFrame, [0, 16], [0, 1], { extrapolateRight: 'clamp' });
  const logoY = interpolate(logoFrame, [0, 16], [24, 0], { extrapolateRight: 'clamp' });

  return (
    <AbsoluteFill>
      <Background />
      <AbsoluteFill style={{ alignItems: 'center', justifyContent: 'center' }}>
        <div
          style={{
            fontSize: 100,
            opacity: moonOpacity,
            transform: `scale(${moonScale})`,
            marginBottom: 18,
          }}
        >
          🌙
        </div>
        <div
          style={{
            opacity: logoOpacity,
            transform: `translateY(${logoY}px)`,
            fontFamily: BRAND.fontTitle,
            fontWeight: 900,
            fontSize: 76,
            color: BRAND.goldLight,
            textAlign: 'center',
            textShadow: '0 6px 30px rgba(0,0,0,0.5)',
          }}
        >
          Pular IA
        </div>
        <div style={{ height: 14 }} />
        <AnimatedSubtitle delay={30} fontSize={30}>
          La langue pular n'a pas encore d'IA.
        </AnimatedSubtitle>
        <AnimatedSubtitle delay={42} fontSize={30} color={BRAND.text}>
          Change ça — avec ta voix.
        </AnimatedSubtitle>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
