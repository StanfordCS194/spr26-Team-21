interface LogoProps {
  className?: string;
}

export default function Logo({ className }: LogoProps) {
  return (
    <span className={`logo-mark ${className || ''}`} aria-hidden="true">
      <span className="logo-f">f</span>
      <span className="logo-slash">/</span>
    </span>
  );
}
