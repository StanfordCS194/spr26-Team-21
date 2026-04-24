import { motion } from 'motion/react';
import { ArrowRight } from '../icons/Icons';

interface PromptBoxProps {
  prompt: string;
  setPrompt: (value: string) => void;
}

export default function PromptBox({ prompt, setPrompt }: PromptBoxProps) {
  return (
    <motion.div
      className="prompt-box"
      initial={{ opacity: 0, filter: 'blur(12px)', y: 12 }}
      animate={{ opacity: 1, filter: 'blur(0px)', y: 0 }}
      transition={{ duration: 0.7, ease: 'easeOut', delay: 0.15 }}
    >
      <textarea
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        placeholder="Describe the dataset you need..."
        aria-label="Describe the dataset you want to synthesize"
      />
      <div className="prompt-box-footer">
        <div />
        <button className="prompt-submit" aria-label="Submit">
          <ArrowRight />
        </button>
      </div>
    </motion.div>
  );
}
