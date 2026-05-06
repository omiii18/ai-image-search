/* ================================================================
   DeepSearch AI — Landing Page
   React 18 + Tailwind CSS + Framer Motion + Lucide
   ================================================================ */

const { useState, useEffect, useRef } = React;
const { motion, AnimatePresence, useInView } = window["framer-motion"] || {};

// ─── Utility: Lucide icon renderer ──────────────────────────────
function LucideIcon({ name, size = 24, color = "currentColor", strokeWidth = 2, className = "" }) {
  const ref = useRef(null);
  useEffect(() => {
    if (ref.current && lucide && lucide.createElement) {
      ref.current.innerHTML = "";
      const el = lucide.createElement(lucide.icons[name]);
      el.setAttribute("width", size);
      el.setAttribute("height", size);
      el.setAttribute("stroke", color);
      el.setAttribute("stroke-width", strokeWidth);
      if (className) el.setAttribute("class", className);
      ref.current.appendChild(el);
    }
  }, [name, size, color, strokeWidth]);
  return React.createElement("span", { ref, className: "inline-flex items-center justify-center " + className });
}

// ─── Fade-in section wrapper ────────────────────────────────────
function FadeSection({ children, className = "", delay = 0 }) {
  const ref = useRef(null);
  const isInView = useInView ? useInView(ref, { once: true, margin: "-80px" }) : true;
  if (!motion) return React.createElement("div", { className, ref }, children);
  return React.createElement(
    motion.div,
    {
      ref,
      className,
      initial: { opacity: 0, y: 50 },
      animate: isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 50 },
      transition: { duration: 0.7, delay, ease: [0.22, 1, 0.36, 1] },
    },
    children
  );
}

// ─── Stagger children ──────────────────────────────────────────
function StaggerContainer({ children, className = "" }) {
  if (!motion) return React.createElement("div", { className }, children);
  return React.createElement(
    motion.div,
    {
      className,
      initial: "hidden",
      whileInView: "visible",
      viewport: { once: true, margin: "-60px" },
      variants: { hidden: {}, visible: { transition: { staggerChildren: 0.15 } } },
    },
    children
  );
}

function StaggerItem({ children, className = "" }) {
  if (!motion) return React.createElement("div", { className }, children);
  return React.createElement(
    motion.div,
    {
      className,
      variants: { hidden: { opacity: 0, y: 30 }, visible: { opacity: 1, y: 0, transition: { duration: 0.6, ease: [0.22, 1, 0.36, 1] } } },
    },
    children
  );
}

// ─── NAVBAR ─────────────────────────────────────────────────────
function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  useEffect(() => {
    const h = () => setScrolled(window.scrollY > 30);
    window.addEventListener("scroll", h);
    return () => window.removeEventListener("scroll", h);
  }, []);

  return React.createElement(
    "nav",
    {
      id: "navbar",
      className: `fixed top-0 left-0 right-0 z-50 transition-all duration-300 font-jakarta ${scrolled ? "bg-offwhite/90 backdrop-blur-md shadow-sm" : "bg-offwhite"}`,
    },
    React.createElement(
      "div",
      { className: "max-w-[1200px] mx-auto flex items-center justify-between px-6 py-4" },
      // Logo
      React.createElement(
        "a",
        { href: "#", className: "text-2xl font-extrabold tracking-tight text-charcoal" },
        "DeepSearch ",
        React.createElement("span", { className: "text-brand" }, "AI")
      ),
      // Links
      React.createElement(
        "div",
        { className: "hidden md:flex items-center gap-8" },
        ["Home", "Features", "FAQ"].map((l) =>
          React.createElement(
            "a",
            {
              key: l,
              href: `#${l.toLowerCase()}`,
              className: "text-sm font-semibold text-charcoal/70 hover:text-brand transition-colors duration-200",
            },
            l
          )
        ),
        React.createElement(
          "a",
          {
            href: "#home",
            className: "ml-2 px-5 py-2.5 bg-brand text-white text-sm font-semibold rounded-lg hover:bg-red-600 transition-all duration-200 hover:scale-105 btn-pulse",
          },
          "Download"
        )
      ),
      // Mobile menu
      React.createElement(
        "button",
        { className: "md:hidden", onClick: () => {} },
        React.createElement(LucideIcon, { name: "menu", size: 24, color: "#1E293B" })
      )
    )
  );
}

// ─── HERO ───────────────────────────────────────────────────────
function Hero() {
  return React.createElement(
    "section",
    { id: "home", className: "section-offwhite pt-28 pb-20 md:pt-36 md:pb-28" },
    React.createElement(
      "div",
      { className: "max-w-[1200px] mx-auto px-6 grid md:grid-cols-2 gap-12 md:gap-16 items-center" },
      // Left
      React.createElement(
        FadeSection,
        { className: "space-y-7" },
        React.createElement(
          "div",
          { className: "inline-flex items-center gap-2 px-4 py-1.5 bg-red-50 border border-red-100 rounded-full" },
          React.createElement(LucideIcon, { name: "sparkles", size: 16, color: "#EF4444" }),
          React.createElement("span", { className: "text-xs font-semibold text-brand tracking-wide uppercase" }, "100% Private & Local")
        ),
        React.createElement(
          "h1",
          { className: "text-4xl sm:text-5xl lg:text-[64px] font-extrabold text-charcoal leading-[1.1] tracking-[-0.02em]" },
          "Search Your Files ",
          React.createElement("br", { className: "hidden lg:block" }),
          "With ",
          React.createElement("span", { className: "text-brand" }, "AI Vision"),
          React.createElement("span", { className: "text-brand" }, ".")
        ),
        React.createElement(
          "p",
          { className: "text-lg md:text-xl text-charcoal/60 leading-relaxed max-w-lg font-medium" },
          "DeepSearch AI understands your photos, documents, and design assets. Find anything instantly using natural language — all processed locally, never leaving your device."
        ),
        React.createElement(
          "div",
          { className: "flex flex-wrap gap-4 pt-2" },
          React.createElement(
            motion ? motion.a : "a",
            {
              href: "#",
              className: "inline-flex items-center gap-2.5 px-7 py-3.5 bg-brand text-white font-semibold rounded-lg hover:bg-red-600 transition-colors btn-pulse",
              whileHover: motion ? { scale: 1.05 } : undefined,
              whileTap: motion ? { scale: 0.97 } : undefined,
            },
            React.createElement(LucideIcon, { name: "apple", size: 20, color: "#fff" }),
            "Download for Mac"
          ),
          React.createElement(
            motion ? motion.a : "a",
            {
              href: "#",
              className: "inline-flex items-center gap-2.5 px-7 py-3.5 bg-charcoal text-white font-semibold rounded-lg hover:bg-slate-800 transition-colors",
              whileHover: motion ? { scale: 1.05 } : undefined,
              whileTap: motion ? { scale: 0.97 } : undefined,
            },
            React.createElement(LucideIcon, { name: "monitor", size: 20, color: "#fff" }),
            "Download for Windows"
          )
        ),
        React.createElement(
          "p",
          { className: "text-xs text-charcoal/40 flex items-center gap-1.5" },
          React.createElement(LucideIcon, { name: "shield-check", size: 14, color: "#94a3b8" }),
          "Free & open source. No data ever leaves your machine."
        )
      ),
      // Right — video placeholder
      React.createElement(
        FadeSection,
        { delay: 0.25 },
        React.createElement(
          motion ? motion.div : "div",
          {
            className: "relative rounded-2xl overflow-hidden shadow-2xl shadow-brand/10 border border-gray-200/50 bg-white aspect-[4/3]",
            whileHover: motion ? { scale: 1.02 } : undefined,
            transition: { duration: 0.4 },
          },
          // Gradient overlay
          React.createElement("div", {
            className: "absolute inset-0 bg-gradient-to-br from-brand/5 via-transparent to-brand/10 z-10 pointer-events-none",
          }),
          // Placeholder content
          React.createElement(
            "div",
            { className: "absolute inset-0 flex flex-col items-center justify-center gap-4 z-20" },
            React.createElement(
              "div",
              { className: "w-16 h-16 rounded-full bg-brand/10 flex items-center justify-center" },
              React.createElement(LucideIcon, { name: "play", size: 28, color: "#EF4444" })
            ),
            React.createElement("p", { className: "text-sm font-medium text-charcoal/50" }, "Product Demo Video")
          ),
          // Animated grid pattern
          React.createElement("div", {
            className: "absolute inset-0 opacity-[0.03]",
            style: {
              backgroundImage: "radial-gradient(circle, #1E293B 1px, transparent 1px)",
              backgroundSize: "20px 20px",
            },
          })
        )
      )
    )
  );
}

// ─── FEATURES (Red Section) ─────────────────────────────────────
const FEATURES = [
  {
    icon: "brain",
    title: "AI-Powered Vision Search",
    desc: "Find any image by describing what's in it naturally.",
    gradient: "from-purple-600 to-pink-500",
  },
  {
    icon: "lock",
    title: "100% Offline & Private",
    desc: "Everything runs locally. Your data never leaves your device.",
    gradient: "from-blue-500 to-cyan-400",
  },
  {
    icon: "zap",
    title: "Blazing Fast Indexing",
    desc: "Index thousands of files in seconds with FAISS vectors.",
    gradient: "from-amber-500 to-orange-400",
  },
];

function FeatureCard({ icon, title, desc, gradient }) {
  return React.createElement(
    StaggerItem,
    {},
    React.createElement(
      "div",
      { className: "glass-card rounded-2xl p-8 md:p-10 cursor-pointer group relative overflow-hidden h-full" },
      // Video/GIF background on hover
      React.createElement("div", {
        className: `card-video-bg absolute inset-0 rounded-2xl bg-gradient-to-br ${gradient} opacity-0 group-hover:opacity-20 transition-opacity duration-500`,
      }),
      React.createElement(
        "div",
        { className: "relative z-10 space-y-4" },
        React.createElement(
          "div",
          { className: "w-14 h-14 rounded-xl bg-white/20 flex items-center justify-center mb-2 group-hover:bg-white/30 transition-colors duration-300" },
          React.createElement(LucideIcon, { name: icon, size: 28, color: "#ffffff" })
        ),
        React.createElement("h3", { className: "text-xl md:text-2xl font-bold text-white" }, title),
        React.createElement("p", { className: "text-white/80 font-medium leading-relaxed" }, desc)
      )
    )
  );
}

function Features() {
  return React.createElement(
    "section",
    { id: "features", className: "section-red section-overlap-top py-24 md:py-32" },
    React.createElement(
      "div",
      { className: "max-w-[1200px] mx-auto px-6" },
      React.createElement(
        FadeSection,
        { className: "text-center mb-16" },
        React.createElement(
          "span",
          { className: "inline-block px-4 py-1.5 bg-white/10 border border-white/20 rounded-full text-xs font-semibold text-white/90 tracking-wide uppercase mb-6" },
          "Core Features"
        ),
        React.createElement(
          "h2",
          { className: "text-3xl sm:text-4xl md:text-[40px] font-bold text-white leading-tight tracking-[-0.01em]" },
          "Everything You Need to",
          React.createElement("br", { className: "hidden sm:block" }),
          " Search Smarter"
        )
      ),
      React.createElement(
        StaggerContainer,
        { className: "grid md:grid-cols-3 gap-6 md:gap-8" },
        FEATURES.map((f) => React.createElement(FeatureCard, { key: f.title, ...f }))
      )
    )
  );
}

// ─── USE CASES (Off-White) ──────────────────────────────────────
const USE_CASES = [
  {
    icon: "palette",
    title: "Find Design Assets Instantly",
    desc: "Stop scrolling through folders. Just type 'that blue gradient banner from last quarter' and DeepSearch finds it.",
    imgGradient: "from-rose-100 to-pink-200",
  },
  {
    icon: "file-text",
    title: "Locate Old Documents Fast",
    desc: "Search through thousands of screenshots, receipts, and scans using the text or visuals you remember.",
    imgGradient: "from-sky-100 to-blue-200",
  },
  {
    icon: "image",
    title: "Organize Photo Libraries",
    desc: "Find 'sunset at the beach' or 'birthday party photos' across your entire library without manual tagging.",
    imgGradient: "from-amber-100 to-yellow-200",
  },
];

function UseCases() {
  return React.createElement(
    "section",
    { className: "section-offwhite section-overlap-top py-24 md:py-32" },
    React.createElement(
      "div",
      { className: "max-w-[1200px] mx-auto px-6" },
      React.createElement(
        FadeSection,
        { className: "text-center mb-16" },
        React.createElement(
          "h2",
          { className: "text-3xl sm:text-4xl md:text-[40px] font-bold text-charcoal leading-tight tracking-[-0.01em]" },
          "Built for non-technical teams",
          React.createElement("br", { className: "hidden sm:block" }),
          " with a need for ",
          React.createElement("span", { className: "text-brand" }, "speed")
        ),
        React.createElement(
          "p",
          { className: "mt-4 text-charcoal/50 text-lg max-w-2xl mx-auto" },
          "No setup. No cloud. Just search."
        )
      ),
      React.createElement(
        "div",
        { className: "space-y-16 md:space-y-24" },
        USE_CASES.map((uc, i) =>
          React.createElement(
            FadeSection,
            { key: uc.title, delay: i * 0.1 },
            React.createElement(
              "div",
              { className: `flex flex-col ${i % 2 === 1 ? "md:flex-row-reverse" : "md:flex-row"} gap-10 md:gap-16 items-center` },
              // Text
              React.createElement(
                "div",
                { className: "flex-1 space-y-5" },
                React.createElement(
                  "div",
                  { className: "w-12 h-12 rounded-xl bg-brand/10 flex items-center justify-center" },
                  React.createElement(LucideIcon, { name: uc.icon, size: 24, color: "#EF4444" })
                ),
                React.createElement("h3", { className: "text-2xl md:text-3xl font-bold text-charcoal" }, uc.title),
                React.createElement("p", { className: "text-charcoal/60 text-lg leading-relaxed" }, uc.desc)
              ),
              // Visual mockup
              React.createElement(
                motion ? motion.div : "div",
                {
                  className: `flex-1 rounded-2xl bg-gradient-to-br ${uc.imgGradient} aspect-[4/3] flex items-center justify-center shadow-lg`,
                  whileHover: motion ? { scale: 1.03 } : undefined,
                  transition: { duration: 0.4 },
                },
                React.createElement(
                  "div",
                  { className: "text-center" },
                  React.createElement(LucideIcon, { name: uc.icon, size: 48, color: "#EF4444" }),
                  React.createElement("p", { className: "mt-3 text-sm font-medium text-charcoal/40" }, "UI Mockup Placeholder")
                )
              )
            )
          )
        )
      )
    )
  );
}

// ─── TECH STACK (Red Section - Box in Box) ──────────────────────
const TECHS = [
  {
    name: "CLIP",
    label: "OpenAI",
    icon: "eye",
    desc: "Vision-Language Model",
  },
  {
    name: "FAISS",
    label: "Meta AI",
    icon: "database",
    desc: "Vector Similarity Search",
  },
  {
    name: "RAG",
    label: "LangChain",
    icon: "workflow",
    desc: "Retrieval Augmented Generation",
  },
];

function TechStack() {
  return React.createElement(
    "section",
    { className: "section-red section-overlap-top py-24 md:py-32" },
    React.createElement(
      "div",
      { className: "max-w-[1200px] mx-auto px-6" },
      React.createElement(
        FadeSection,
        { className: "text-center mb-14" },
        React.createElement(
          "span",
          { className: "inline-block px-4 py-1.5 bg-white/10 border border-white/20 rounded-full text-xs font-semibold text-white/90 tracking-wide uppercase mb-6" },
          "Powered By"
        ),
        React.createElement(
          "h2",
          { className: "text-3xl sm:text-4xl md:text-[40px] font-bold text-white leading-tight tracking-[-0.01em]" },
          "Built on World-Class AI"
        )
      ),
      // Outer Box
      React.createElement(
        FadeSection,
        { delay: 0.15 },
        React.createElement(
          "div",
          { className: "border-2 border-white/20 rounded-3xl p-6 md:p-10 bg-white/5 backdrop-blur-sm" },
          React.createElement(
            StaggerContainer,
            { className: "grid md:grid-cols-3 gap-6" },
            TECHS.map((t) =>
              React.createElement(
                StaggerItem,
                { key: t.name },
                React.createElement(
                  "div",
                  { className: "tech-card bg-white rounded-2xl p-8 text-center space-y-4 hover:shadow-xl hover:-translate-y-1 transition-all duration-300 cursor-pointer group" },
                  React.createElement(
                    "div",
                    { className: "w-16 h-16 mx-auto rounded-2xl bg-brand/10 flex items-center justify-center group-hover:bg-brand/15 transition-colors" },
                    React.createElement(LucideIcon, { name: t.icon, size: 32, color: "#EF4444" })
                  ),
                  React.createElement("h3", { className: "text-2xl font-bold text-charcoal icon-shimmer" }, t.name),
                  React.createElement("p", { className: "text-sm font-semibold text-brand" }, t.label),
                  React.createElement("p", { className: "text-sm text-charcoal/50" }, t.desc)
                )
              )
            )
          )
        )
      )
    )
  );
}

// ─── FAQ (Off-White) ────────────────────────────────────────────
const FAQS = [
  { q: "Is DeepSearch AI really 100% offline?", a: "Yes! All AI models run entirely on your local machine. No data is ever sent to a cloud server. Your files, your privacy." },
  { q: "What file types does it support?", a: "DeepSearch AI currently supports all major image formats (JPG, PNG, WEBP, BMP, GIF) and is expanding to PDFs and documents." },
  { q: "How fast is the indexing?", a: "FAISS-powered vector indexing can process thousands of images in under a minute on most modern hardware with GPU acceleration." },
  { q: "Does it work on both Mac and Windows?", a: "Yes! We provide native builds for both macOS (Apple Silicon & Intel) and Windows (x64). Linux support is coming soon." },
  { q: "Is it free?", a: "DeepSearch AI is completely free and open source under the MIT license. No subscriptions, no hidden fees." },
];

function FAQItem({ q, a, isOpen, toggle }) {
  return React.createElement(
    "div",
    {
      className: "border border-gray-200/80 rounded-xl overflow-hidden transition-all duration-200 hover:border-brand/30 bg-white",
    },
    React.createElement(
      "button",
      {
        onClick: toggle,
        className: "w-full flex items-center justify-between px-6 py-5 text-left",
      },
      React.createElement("span", { className: "font-semibold text-charcoal pr-4" }, q),
      React.createElement(
        motion ? motion.div : "div",
        {
          animate: motion ? { rotate: isOpen ? 180 : 0 } : undefined,
          transition: { duration: 0.3 },
          className: "flex-shrink-0",
        },
        React.createElement(LucideIcon, { name: "chevron-down", size: 20, color: "#EF4444" })
      )
    ),
    React.createElement(
      "div",
      { className: `accordion-content ${isOpen ? "open" : ""}` },
      React.createElement("p", { className: "px-6 pb-5 text-charcoal/60 leading-relaxed" }, a)
    )
  );
}

function FAQ() {
  const [openIdx, setOpenIdx] = useState(0);
  return React.createElement(
    "section",
    { id: "faq", className: "section-offwhite section-overlap-top py-24 md:py-32" },
    React.createElement(
      "div",
      { className: "max-w-[700px] mx-auto px-6" },
      React.createElement(
        FadeSection,
        { className: "text-center mb-14" },
        React.createElement(
          "h2",
          { className: "text-3xl sm:text-4xl md:text-[40px] font-bold text-charcoal tracking-[-0.01em]" },
          "Frequently Asked Questions"
        )
      ),
      React.createElement(
        StaggerContainer,
        { className: "space-y-3" },
        FAQS.map((f, i) =>
          React.createElement(
            StaggerItem,
            { key: i },
            React.createElement(FAQItem, { q: f.q, a: f.a, isOpen: openIdx === i, toggle: () => setOpenIdx(openIdx === i ? -1 : i) })
          )
        )
      )
    )
  );
}

// ─── FOOTER ─────────────────────────────────────────────────────
function Footer() {
  return React.createElement(
    "footer",
    { className: "bg-offwhite border-t border-gray-200/60 py-10" },
    React.createElement(
      "div",
      { className: "max-w-[1200px] mx-auto px-6 flex flex-col sm:flex-row items-center justify-between gap-4" },
      React.createElement(
        "p",
        { className: "text-sm text-charcoal/40" },
        "© 2026 DeepSearch AI. All rights reserved."
      ),
      React.createElement(
        "div",
        { className: "flex items-center gap-5" },
        [
          { icon: "github", href: "#" },
          { icon: "twitter", href: "#" },
          { icon: "mail", href: "mailto:hello@deepsearchai.app" },
        ].map((l) =>
          React.createElement(
            "a",
            {
              key: l.icon,
              href: l.href,
              className: "text-charcoal/40 hover:text-brand transition-colors duration-200",
            },
            React.createElement(LucideIcon, { name: l.icon, size: 20 })
          )
        )
      )
    )
  );
}

// ─── APP ────────────────────────────────────────────────────────
function App() {
  return React.createElement(
    "div",
    { className: "font-jakarta" },
    React.createElement(Navbar),
    React.createElement(Hero),
    React.createElement(Features),
    React.createElement(UseCases),
    React.createElement(TechStack),
    React.createElement(FAQ),
    React.createElement(Footer)
  );
}

// ─── RENDER ─────────────────────────────────────────────────────
const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(React.createElement(App));
