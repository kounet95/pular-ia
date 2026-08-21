import React from 'react';
import { AbsoluteFill, useCurrentFrame, useVideoConfig, interpolate, spring } from 'remotion';
import { BRAND } from '../brand';
import { Background } from './Background';
import { AnimatedTitle, AnimatedSubtitle } from './AnimatedTitle';

const Counter: React.FC<{ to: number; delay: number; label: string; suffix?: string }> = ({
  to,
  delay,
  label,
  suffix = '',
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const local = Math.max(0, frame - delay);
  const progress = spring({ frame: local, fps, config: { damping: 200 }, durationInFrames: 45 });
  const value = Math.round(interpolate(progress, [0, 1], [0, to]));
  const opacity = interpolate(local, [0, 10], [0, 1], { extrapolateRight: 'clamp' });

  return (
    <div style={{ opacity, textAlign: 'center', minWidth: 220 }}>
      <div
        style={{
          fontFamily: BRAND.fontTitle,
          fontWeight: 800,
          fontSize: 76,
          color: BRAND.gold,
        }}
      >
        {value.toLocaleString('fr-FR')}
        {suffix}
      </div>
      <div style={{ fontFamily: BRAND.fontBody, fontSize: 26, color: BRAND.gris, marginTop: 6 }}>
        {label}
      </div>
    </div>
  );
};

export const StatsScene: React.FC = () => {
  return (
    <AbsoluteFill>
      <Background variant="gold" />
      <AbsoluteFill style={{ alignItems: 'center', justifyContent: 'center', padding: '0 60px' }}>
        <AnimatedTitle delay={0} fontSize={54}>
          Une communauté qui grandit chaque jour
        </AnimatedTitle>
        <div style={{ height: 60 }} />
        <div style={{ display: 'flex', gap: 70 }}>
          <Counter to={12480} delay={18} label="Contributions" />
          <Counter to={3190} delay={26} label="Contributeurs" />
        </div>
        <div style={{ height: 40 }} />
        <AnimatedSubtitle delay={45} fontSize={28}>
          Chaque voix compte. La tienne peut être la suivante.
        </AnimatedSubtitle>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
