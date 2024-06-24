\ProvidesClass{ipppcompcourse}
\LoadClass[a4paper,11pt,twoside]{article}
\RequirePackage{a4wide,xspace,setspace,verbatim}
\RequirePackage{ipppcompcourse}
\setlength{\parindent}{0cm}
\onehalfspacing

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\def\lecturename{foo}
\def\lecturesheet{}
\def\lecturedate{\today}
\DeclareRobustCommand{\setlecturename}[1]{\def\lecturename{#1}}
\DeclareRobustCommand{\setlecturesheet}[1]{\def\lecturesheet{#1}}
\DeclareRobustCommand{\setlecturedate}[1]{\def\lecturedate{#1}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\AtBeginDocument{
  Summer school\hfill Tutorial
  session 1
  \newline
  \hfill Herwig++ examples\newline
  \lecturedate \newline
  \rule{\textwidth}{.5mm}
  \begin{center}
    \textbf{\LARGE \lecturename}%\\
  \end{center}
  \setlength{\parskip}{3pt} 
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%\DeclareRobustCommand{\exercise}[1]{\begin{center}\ensuremath{\blacktriangleright}\hspace{1em}\textbf{#1}\hspace{1em}\ensuremath{\blacktriangleleft}\end{center}}

\DeclareRobustCommand{\exercise}[1]{%
\begin{center}\ensuremath{\blacktriangleright}\hspace{1em}
\begin{minipage}[t]{0.7\columnwidth}
\textbf{#1}
\end{minipage}
\hspace{1em}\ensuremath{\blacktriangleleft}
\end{center}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\setlength{\fboxrule}{1.5pt}
\DeclareRobustCommand{\timeforbreak}[1]{%
\vspace{0.5cm}
\fbox{
\begin{minipage}[t][3\height][c]{0.965\columnwidth}
\begin{center}
\textbf{
\large
#1
}
\end{center}
\end{minipage}
}
}

\DeclareRobustCommand{\timeforabreak}[1]{\timeforbreak{#1}}
\DeclareRobustCommand{\timefortea}[1]{\timeforbreak{#1}}

\DeclareRobustCommand{\timefornotes}[1]{%
\vspace{0.5cm}
\fbox{
\begin{minipage}[t][6\height][t]{0.965\columnwidth}
\vspace{\baselineskip}
\begin{center}
\textbf{
\large
#1
}
\end{center}
\end{minipage}
}
}

\DeclareRobustCommand{\codelisting}[1]{%
%\newline
\vspace{0.6cm}
\hrule height 1.5pt width 0.75\textwidth
\verbatiminput{#1}
%\vspace{-0.2cm}
\hrule height 1.5pt width 0.75\textwidth
\vspace{0.5cm}
}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


