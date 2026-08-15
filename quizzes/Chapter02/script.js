import quizData from './questions.js';

let currentQuestion = 0;
// userAnswers[i] will store state: { wrongSelections: [], isCorrect: boolean, ... }
let userAnswers = new Array(quizData.length).fill(null);

// DOM Elements
const progressText = document.getElementById('progress-text');
const progressBar = document.getElementById('progress-bar');
const questionArea = document.getElementById('question-area');
const explanationBox = document.getElementById('explanation-box');
const explanationText = document.getElementById('explanation-text');
const prevBtn = document.getElementById('prev-btn');
const checkBtn = document.getElementById('check-btn');
const nextBtn = document.getElementById('next-btn');
const resultScreen = document.getElementById('result-screen');
const scoreCircle = document.getElementById('score-circle');
const scoreText = document.getElementById('score-text');
const quizMain = document.getElementById('quiz-main');

function getAnswerState(idx) {
  if (!userAnswers[idx]) {
    userAnswers[idx] = { wrongSelections: [], isCorrect: false };
  }
  return userAnswers[idx];
}

function loadQuestion() {
  explanationBox.classList.remove('show');
  
  const q = quizData[currentQuestion];
  const state = getAnswerState(currentQuestion);
  const isFullyAnswered = state.isCorrect;

  // Setup buttons
  prevBtn.style.display = currentQuestion === 0 ? 'none' : 'inline-block';
  nextBtn.style.display = 'inline-block'; // Always visible to allow skipping
  nextBtn.innerText = currentQuestion === quizData.length - 1 ? "Xem kết quả" : "Câu sau >";
  
  if (isFullyAnswered) {
      checkBtn.style.display = 'none';
  } else {
      if (q.type === 'mcq') {
          checkBtn.style.display = 'none';
      } else {
          checkBtn.style.display = 'inline-block';
          checkBtn.disabled = true;
      }
  }
  
  // Update Header
  progressText.innerText = `Câu hỏi ${currentQuestion + 1} / ${quizData.length}`;
  progressBar.style.width = `${((currentQuestion) / quizData.length) * 100}%`;
  
  // Render Question
  let diffClass = q.difficulty === 'Dễ' ? 'diff-dễ' : (q.difficulty === 'Trung bình' ? 'diff-trung' : 'diff-khó');
  
  let html = `
    <span class="difficulty-badge ${diffClass}">${q.difficulty}</span>
    <div class="question-text">${q.question}</div>
  `;
  
  // Render based on type
  if (q.type === 'mcq') {
    html += `<div class="options-grid">`;
    q.options.forEach((opt, index) => {
      html += `<div class="option" data-index="${index}">${opt}</div>`;
    });
    html += `</div>`;
  } 
  else if (q.type === 'fill_blank') {
    html += `
      <div class="fill-blank-container">
        <input type="text" id="blank-input" class="fill-blank-input" placeholder="Nhập câu trả lời của bạn..." autocomplete="off">
      </div>
    `;
  }
  else if (q.type === 'sorting') {
    html += `<ul id="sortable-list" class="draggable-list">`;
    let stepsToRender = q.steps;
    if (state.steps) stepsToRender = state.steps;
    else {
      stepsToRender = [...q.steps].sort(() => Math.random() - 0.5);
    }
    
    stepsToRender.forEach((step) => {
      let origIndex = q.steps.indexOf(step);
      html += `<li class="draggable-item" data-id="${origIndex}">
        <span style="margin-right:10px; color:#9CA3AF;">☰</span> ${step}
      </li>`;
    });
    html += `</ul>`;
  }
  else if (q.type === 'matching') {
    let rightItems = [...q.pairs].map(p => p.right).sort();
    html += `<div class="matching-container">`;
    q.pairs.forEach((pair, index) => {
      html += `
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
          <div style="flex:1; padding:12px; background:#f3f4f6; border-radius:8px;">${pair.left}</div>
          <select id="match-${index}" class="fill-blank-input" style="flex:1; margin-bottom:0;">
            <option value="" disabled selected>-- Chọn đáp án tương ứng --</option>
            ${rightItems.map(r => `<option value="${r}">${r}</option>`).join('')}
          </select>
        </div>
      `;
    });
    html += `</div>`;
  }
  
  questionArea.innerHTML = html;
  
  // Attach events and restore state
  if (q.type === 'mcq') {
    const options = document.querySelectorAll('.option');
    
    // Restore previous wrong guesses
    state.wrongSelections.forEach(idx => {
      options[idx].classList.add('wrong');
    });

    if (isFullyAnswered) {
      options[q.correctAnswer].classList.add('correct');
    } else {
      options.forEach(opt => {
        opt.addEventListener('click', () => {
          if (state.isCorrect) return; // Prevent clicking after correct
          
          const selectedIndex = parseInt(opt.getAttribute('data-index'));
          
          if (state.wrongSelections.includes(selectedIndex)) return; // Ignore if already clicked wrong

          if (selectedIndex === q.correctAnswer) {
            state.isCorrect = true;
            // Record points only if no wrong attempts were made
            state.earnedPoint = state.wrongSelections.length === 0;
            loadQuestion(); // Reload to show correct state and explanation
          } else {
            state.wrongSelections.push(selectedIndex);
            opt.classList.add('wrong');
          }
        });
      });
    }
  } 
  else if (q.type === 'fill_blank') {
    const input = document.getElementById('blank-input');
    if (state.value !== undefined) {
      input.value = state.value;
      input.disabled = true;
      input.style.borderColor = state.isCorrect ? 'var(--secondary)' : 'var(--danger)';
      input.style.backgroundColor = state.isCorrect ? '#D1FAE5' : '#FEE2E2';
    } else {
      input.addEventListener('input', () => {
        checkBtn.disabled = input.value.trim() === '';
      });
    }
  }
  else if (q.type === 'sorting') {
    if (state.steps !== undefined) {
       const listItems = document.querySelectorAll('.draggable-item');
       listItems.forEach((item, index) => {
          const id = parseInt(item.getAttribute('data-id'));
          if (id === index) {
            item.style.borderColor = 'var(--secondary)';
            item.style.backgroundColor = '#D1FAE5';
          } else {
            item.style.borderColor = 'var(--danger)';
            item.style.backgroundColor = '#FEE2E2';
          }
       });
    } else {
      new Sortable(document.getElementById('sortable-list'), { animation: 150 });
      checkBtn.disabled = false; 
    }
  }
  else if (q.type === 'matching') {
    if (state.selections !== undefined) {
      q.pairs.forEach((pair, index) => {
        const sel = document.getElementById(`match-${index}`);
        sel.value = state.selections[index];
        sel.disabled = true;
        if (sel.value === pair.right) {
          sel.style.borderColor = 'var(--secondary)';
          sel.style.backgroundColor = '#D1FAE5';
        } else {
          sel.style.borderColor = 'var(--danger)';
          sel.style.backgroundColor = '#FEE2E2';
        }
      });
    } else {
      const selects = document.querySelectorAll('select');
      selects.forEach(s => s.addEventListener('change', () => {
        let allSelected = Array.from(selects).every(sel => sel.value !== "");
        checkBtn.disabled = !allSelected;
      }));
    }
  }

  // Only show explanation if they finally got it correct
  // (For other types, checking the answer marks it as "attempted" and shows explanation regardless of right/wrong)
  // For MCQ, we force them to click until correct. 
  // Let's show explanation if isCorrect is true, OR if they've submitted an answer for non-MCQ.
  if (isFullyAnswered || (q.type !== 'mcq' && state.attempted)) {
    explanationText.innerText = q.explanation;
    explanationBox.classList.add('show');
  }
}

function checkAnswer() {
  const q = quizData[currentQuestion];
  const state = getAnswerState(currentQuestion);
  
  if (q.type === 'fill_blank') {
    const input = document.getElementById('blank-input');
    const val = input.value.trim().toLowerCase();
    state.isCorrect = q.blanks.some(b => val.includes(b.toLowerCase()) || b.toLowerCase().includes(val));
    state.value = input.value;
    state.earnedPoint = state.isCorrect;
    state.attempted = true;
  }
  else if (q.type === 'sorting') {
    const listItems = document.querySelectorAll('.draggable-item');
    state.isCorrect = true;
    let currentSteps = [];
    listItems.forEach((item, index) => {
      currentSteps.push(item.innerText.replace('☰', '').trim());
      const id = parseInt(item.getAttribute('data-id'));
      if (id !== index) state.isCorrect = false;
    });
    state.steps = currentSteps;
    state.earnedPoint = state.isCorrect;
    state.attempted = true;
  }
  else if (q.type === 'matching') {
    state.isCorrect = true;
    let selections = [];
    q.pairs.forEach((pair, index) => {
      const sel = document.getElementById(`match-${index}`);
      selections.push(sel.value);
      if (sel.value !== pair.right) state.isCorrect = false;
    });
    state.selections = selections;
    state.earnedPoint = state.isCorrect;
    state.attempted = true;
  }
  
  // Reload question to show styling and explanation
  loadQuestion();
}

function showResult() {
  quizMain.style.display = 'none';
  resultScreen.classList.add('show');
  progressBar.style.width = '100%';
  progressText.innerText = "Hoàn thành";
  
  let score = userAnswers.filter(a => a && a.earnedPoint).length;
  
  scoreText.innerText = `${score}/${quizData.length}`;
  const percentage = (score / quizData.length) * 100;
  scoreCircle.style.background = `conic-gradient(var(--secondary) ${percentage}%, #E5E7EB 0%)`;
}

checkBtn.addEventListener('click', () => {
  checkAnswer();
});

nextBtn.addEventListener('click', () => {
  if (currentQuestion < quizData.length - 1) {
    currentQuestion++;
    loadQuestion();
  } else {
    showResult();
  }
});

prevBtn.addEventListener('click', () => {
  if (currentQuestion > 0) {
    currentQuestion--;
    loadQuestion();
  }
});

// Initialize
loadQuestion();
