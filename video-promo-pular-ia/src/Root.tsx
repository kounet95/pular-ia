import React from 'react';
import { Composition } from 'remotion';
import { PularIAPromo, PULAR_IA_PROMO_DURATION } from './compositions/PularIAPromo';

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="PularIAPromo"
        component={PularIAPromo}
        durationInFrames={PULAR_IA_PROMO_DURATION}
        fps={30}
        width={1080}
        height={1920}
      />
      <Composition
        id="PularIAPromoSquare"
        component={PularIAPromo}
        durationInFrames={PULAR_IA_PROMO_DURATION}
        fps={30}
        width={1080}
        height={1080}
      />
      <Composition
        id="PularIAPromoHorizontal"
        component={PularIAPromo}
        durationInFrames={PULAR_IA_PROMO_DURATION}
        fps={30}
        width={1920}
        height={1080}
      />
    </>
  );
};
