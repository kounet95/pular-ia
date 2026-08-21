import React from 'react';
import { AbsoluteFill, useCurrentFrame, interpolate } from 'remotion';
import { BRAND } from '../brand';
import { Background } from './Background';
import { AnimatedTitle, AnimatedSubtitle } from './AnimatedTitle';

export const InfluencerCallout: React.FC = () => {
  const frame = useCurrentFrame();

  const pulse = 1 + Math.sin(frame / 8) * 0.03;
  const chipOpacity = interpolate(frame, [40, 55], [0, 1], { extrapolateRight: 'clamp' });

  return (
    <AbsoluteFill>
      <Background variant="magenta" />
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
            marginBottom: 24,
            transform: `scale(${pulse})`,
          }}
        >
          📱🎤
        </div>
        <AnimatedTitle delay={0} fontSize={58} color={BRAND.goldLight}>
          Toi qui as une communauté...
        </AnimatedTitle>
        <div style={{ height: 18 }} />
        <AnimatedTitle delay={12} fontSize={44}>
          Montre-la. Filme-toi. Parles-en.
        </AnimatedTitle>
        <div style={{ height: 26 }} />
        <AnimatedSubtitle delay={28} fontSize={30}>
          Créateurs, influenceurs peuls et africains : enregistrez-vous en train de
          contribuer, taguez Pular IA, et faites entrer votre communauté dans l'histoire
          de la langue pular.
        </AnimatedSubtitle>
        <div
          style={{
            opacity: chipOpacity,
            marginTop: 34,
            fontFamily: BRAND.fontBody,
            fontWeight: 700,
            fontSize: 24,
            color: BRAND.bgDark,
            background: BRAND.goldLight,
            borderRadius: 999,
            padding: '12px 30px',
          }}
        >
          #PularIA · #FuutaJaloo
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
