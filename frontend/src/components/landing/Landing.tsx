import { useState } from 'react';
import { motion } from 'motion/react';
import Logo from '../Logo';
import PromptBox from './PromptBox';

export default function Landing() {
  const [prompt, setPrompt] = useState('');

  return (
    <div className="landing">
      <motion.h1
        className="landing-title"
        initial={{ opacity: 0, filter: 'blur(12px)', y: 8 }}
        animate={{ opacity: 1, filter: 'blur(0px)', y: 0 }}
        transition={{ duration: 0.7, ease: 'easeOut' }}
      >
        <Logo className="landing-title-logo" />
        What should we synthesize?
      </motion.h1>

      <PromptBox prompt={prompt} setPrompt={setPrompt} />
    </div>
  );
}
