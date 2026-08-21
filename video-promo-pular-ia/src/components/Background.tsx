import React from 'react';
import { AbsoluteFill, useCurrentFrame, useVideoConfig, interpolate } from 'remotion';
import { BRAND } from '../brand';

export const Background: React.FC<{ variant?: 'default' | 'gold' | 'magenta' }> = ({
  variant = 'default',
}) => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();

  const drift = interpolate(frame, [0, durationInFrames], [0, 60]);

  const glow =
    variant === 'gold' ? BRAND.gold : variant === 'magenta' ? BRAND.magenta : BRAND.greenLight;

  return (
    <AbsoluteFill
      style={{
        background: `radial-gradient(circle at 50% 0%, ${BRAND.green} 0%, ${BRAND.bgDark} 60%)`,
      }}
    >
      {/* Moon glow top */}
      <div
        style={{
          position: 'absolute',
          top: -180 + drift * 0.2,
          left: '50%',
          transform: 'translateX(-50%)',
          width: 900,
          height: 900,
          borderRadius: '50%',
          background: `radial-gradient(circle, ${glow}55 0%, transparent 70%)`,
          filter: 'blur(10px)',
        }}
      />
      {/* Subtle stars */}
      {Array.from({ length: 28 }).map((_, i) => {
        const seed = i * 137.5;
        const x = (seed * 3.7) % 100;
        const y = (seed * 5.3) % 100;
        const twinkle = Math.abs(Math.sin((frame + i * 12) / 20));
        return (
          <div
            key={i}
            style={{
              position: 'absolute',
              left: `${x}%`,
              top: `${y}%`,
              width: 3,
              height: 3,
              borderRadius: '50%',
              background: BRAND.goldLight,
              opacity: 0.15 + twinkle * 0.5,
            }}
          />
        );
      })}
      {/* Bottom vignette */}
      <div
        style={{
          position: 'absolute',
          bottom: 0,
          left: 0,
          right: 0,
          height: '40%',
          background: `linear-gradient(to top, ${BRAND.bgDark} 0%, transparent 100%)`,
        }}
      />
    </AbsoluteFill>
  );
};
