import React from 'react';
import { AbsoluteFill, Sequence } from 'remotion';
import { IntroScene } from '../components/IntroScene';
import { FeatureScene } from '../components/FeatureScene';
import { StatsScene } from '../components/StatsScene';
import { InfluencerCallout } from '../components/InfluencerCallout';
import { DonationCTA } from '../components/DonationCTA';
import { OutroScene } from '../components/OutroScene';

const FEATURES: {
  emoji: string;
  title: string;
  description: string;
  badge: string;
  accent: 'gold' | 'magenta';
}[] = [
  {
    emoji: '🎙️',
    title: 'Enregistre ta voix',
    description: 'Parle en pular, 5 à 60 secondes. Une phrase suggérée ou libre — comme tu veux.',
    badge: 'Étape 1',
    accent: 'gold',
  },
  {
    emoji: '📝',
    title: 'Transcription automatique',
    description: "L'IA transcrit ta voix en texte. Tu corriges si besoin, en un instant.",
    badge: 'Étape 2',
    accent: 'magenta',
  },
  {
    emoji: '✅',
    title: 'Validation communautaire',
    description: 'Chaque contribution validée enrichit le corpus qui entraîne le futur modèle pular.',
    badge: 'Étape 3',
    accent: 'gold',
  },
  {
    emoji: '📚',
    title: 'Livres & Librairie',
    description: 'Ajoute livres, poèmes, articles en pular — ou vends tes livres et sois payé automatiquement.',
    badge: 'Fonctionnalité',
    accent: 'magenta',
  },
  {
    emoji: '🏛️',
    title: 'Histoire & Patrimoine peul',
    description: 'Un espace dédié pour préserver et transmettre la mémoire du Fuuta Jaloo.',
    badge: 'Fonctionnalité',
    accent: 'gold',
  },
  {
    emoji: '📰',
    title: 'Espace Éditorial',
    description: 'Écris et publie tes éditos. Ta plume compte autant que ta voix.',
    badge: 'Fonctionnalité',
    accent: 'magenta',
  },
  {
    emoji: '🎮',
    title: 'Quiz Live & Duels',
    description: 'Apprends le pular en t’amusant, seul ou en duel contre la communauté.',
    badge: 'Fonctionnalité',
    accent: 'gold',
  },
  {
    emoji: '🎓',
    title: 'Espace Créateur / Professeur',
    description: 'Un espace pensé pour les enseignants et créateurs de contenu pédagogique.',
    badge: 'Fonctionnalité',
    accent: 'magenta',
  },
  {
    emoji: '👤',
    title: 'Rejoins en 30 secondes',
    description: 'Compte email ou connexion instantanée avec Telegram. Aucune friction.',
    badge: 'Fonctionnalité',
    accent: 'gold',
  },
];

const INTRO = 90;
const FEATURE_DUR = 80;
const STATS = 90;
const INFLUENCER = 150;
const DONATION = 130;
const OUTRO = 130;

export const PularIAPromo: React.FC = () => {
  let cursor = 0;
  const introStart = cursor;
  cursor += INTRO;
  const featuresStart = cursor;
  cursor += FEATURE_DUR * FEATURES.length;
  const statsStart = cursor;
  cursor += STATS;
  const influencerStart = cursor;
  cursor += INFLUENCER;
  const donationStart = cursor;
  cursor += DONATION;
  const outroStart = cursor;
  cursor += OUTRO;

  return (
    <AbsoluteFill>
      <Sequence from={introStart} durationInFrames={INTRO}>
        <IntroScene />
      </Sequence>

      {FEATURES.map((f, i) => (
        <Sequence
          key={f.title}
          from={featuresStart + i * FEATURE_DUR}
          durationInFrames={FEATURE_DUR}
        >
          <FeatureScene
            emoji={f.emoji}
            title={f.title}
            description={f.description}
            badge={f.badge}
            accent={f.accent}
          />
        </Sequence>
      ))}

      <Sequence from={statsStart} durationInFrames={STATS}>
        <StatsScene />
      </Sequence>

      <Sequence from={influencerStart} durationInFrames={INFLUENCER}>
        <InfluencerCallout />
      </Sequence>

      <Sequence from={donationStart} durationInFrames={DONATION}>
        <DonationCTA />
      </Sequence>

      <Sequence from={outroStart} durationInFrames={OUTRO}>
        <OutroScene />
      </Sequence>
    </AbsoluteFill>
  );
};

export const PULAR_IA_PROMO_DURATION = INTRO + FEATURE_DUR * FEATURES.length + STATS + INFLUENCER + DONATION + OUTRO;
