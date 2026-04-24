import './App.css';
import './styles/landing.css';
import Logo from './components/Logo';
import Landing from './components/landing/Landing';

function App() {
  return (
    <div className="app">
      <header className="header">
        <div className="header-left" aria-hidden="true">
          <Logo className="header-logo" />
          <span>Aperture</span>
        </div>
      </header>

      <Landing />
    </div>
  );
}

export default App;
