import "./styles.css";
import {
  BrowserRouter as Router,
  Routes,
  Route,
} from "react-router-dom";
import Home from './components/home'
import About from './components/about'

function App() {
	document.body.style.backgroundImage = `url(http://${window.location.hostname}/static/banner.jpg)`;
  return (
    <>
    <Router>
      <Routes>
        <Route exact path="/" element={<Home/>} />
        <Route path="/about" element={<About/>} />
      </Routes>
    </Router>
    </>
  );
};
export default App;
