let currentSessionId = null;
let eventSource = null;
let currentResult = null;

// API base URL - automatically detects if running on different port
// If page is served from port other than 5000, use localhost:5000 for API
const API_BASE_URL = (() => {
    const origin = window.location.origin;
    const port = window.location.port;
    // If port is 5000 or no port (default), use same origin
    // Otherwise, assume FastAPI is on port 5000
    if (!port || port === '5000') {
        return origin;
    }
    // Extract protocol and hostname, use port 5000
    return `${window.location.protocol}//${window.location.hostname}:5000`;
})();

// Example JSON input
const exampleInput = {
    "pathogens": [
        {
            "pathogen_name": "Staphylococcus aureus",
            "pathogen_count": "10^3 CFU/ML"
        },
        {
            "pathogen_name": "Enterococcus faecalis",
            "pathogen_count": "10^4 CFU/ML"
        }
    ],
    "resistant_genes": [
        "mecA",
        "tetM",
        "dfrA",
        "Ant-la"
    ],
    "severity_codes": [
        "A41.2",
        "A41.81"
    ],
    "age": 32,
    "sample": "Blood",
    "systemic": true
};

// DOM elements
const jsonInput = document.getElementById('jsonInput');
const loadExampleBtn = document.getElementById('loadExample');
const runPipelineBtn = document.getElementById('runPipeline');
const progressSection = document.getElementById('progressSection');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const progressLog = document.getElementById('progressLog');
const resultSection = document.getElementById('resultSection');
const resultContent = document.getElementById('resultContent');
const downloadJsonBtn = document.getElementById('downloadJson');
const downloadPdfBtn = document.getElementById('downloadPdf');
const viewJsonBtn = document.getElementById('viewJson');
const refreshSessionsBtn = document.getElementById('refreshSessions');
const sessionsTableBody = document.getElementById('sessionsTableBody');

// Load example
loadExampleBtn.addEventListener('click', () => {
    jsonInput.value = JSON.stringify(exampleInput, null, 2);
    addLog('Example JSON loaded', 'success');
});

// Run pipeline
runPipelineBtn.addEventListener('click', async () => {
    // Validate JSON first
    let inputData;
    try {
        inputData = JSON.parse(jsonInput.value);
    } catch (e) {
        alert('Invalid JSON. Please fix the JSON format.');
        return;
    }

    // Disable button
    runPipelineBtn.disabled = true;
    runPipelineBtn.textContent = 'Running...';

    // Reset UI
    progressSection.style.display = 'block';
    resultSection.style.display = 'none';
    progressLog.innerHTML = '';
    updateProgress(0, 'Starting pipeline...');

    try {
        // Start pipeline
        const response = await fetch(`${API_BASE_URL}/api/run`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(inputData)
        });

        if (!response.ok) {
            const errorData = await response.json();
            if (response.status === 409 && errorData.current_session_id) {
                // Pipeline already running
                const confirmResume = confirm('A pipeline is already running. Would you like to view its progress?');
                if (confirmResume) {
                    resumeSession(errorData.current_session_id);
                }
                runPipelineBtn.disabled = false;
                runPipelineBtn.textContent = 'Run Pipeline';
                return;
            }
            throw new Error(errorData.error || 'Failed to start pipeline');
        }

        const data = await response.json();
        currentSessionId = data.session_id;

        // Connect to SSE stream (sessions table will refresh on completion/error)
        connectToProgressStream(currentSessionId);

    } catch (error) {
        addLog('Error: ' + error.message, 'error');
        updateProgress(0, 'Error occurred');
        runPipelineBtn.disabled = false;
        runPipelineBtn.textContent = 'Run Pipeline';
    }
});

// Connect to SSE progress stream
function connectToProgressStream(sessionId) {
    if (eventSource) {
        eventSource.close();
    }

    eventSource = new EventSource(`${API_BASE_URL}/api/progress/${sessionId}`);

    eventSource.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);

            if (data.error) {
                addLog('Error: ' + data.error, 'error');
                updateProgress(0, 'Error occurred');
                eventSource.close();
                eventSource = null;
                runPipelineBtn.disabled = false;
                runPipelineBtn.textContent = 'Run Pipeline';
                // Refresh sessions table only once after error
                setTimeout(() => loadSessions(), 500);
                return;
            }

            if (data.result) {
                // Pipeline completed
                currentResult = data.result;
                displayResults(data.result);
                updateProgress(100, 'Pipeline completed!');
                addLog('✓ Pipeline completed successfully', 'success');
                eventSource.close();
                eventSource = null;
                runPipelineBtn.disabled = false;
                runPipelineBtn.textContent = 'Run Pipeline';
                // Refresh sessions table only once after completion
                setTimeout(() => loadSessions(), 500);
                return;
            }

            // Progress update
            const progress = data.progress || 0;
            const stage = data.stage || 'unknown';
            const message = data.message || 'Processing...';

            updateProgress(progress, message);
            addLog(`[${stage}] ${message}`);

        } catch (e) {
            console.error('Error parsing SSE data:', e);
        }
    };

    eventSource.onerror = (error) => {
        console.error('SSE error:', error);
        addLog('Connection error. Retrying...', 'error');
        // Could implement retry logic here
    };
}

// Update progress bar
function updateProgress(percent, message) {
    progressFill.style.width = percent + '%';
    progressText.textContent = percent + '%';
    if (message) {
        addLog(message);
    }
}

// Add log entry
function addLog(message, type = '') {
    const entry = document.createElement('div');
    entry.className = 'log-entry' + (type ? ' ' + type : '');
    entry.textContent = new Date().toLocaleTimeString() + ' - ' + message;
    progressLog.appendChild(entry);
    progressLog.scrollTop = progressLog.scrollHeight;
}

// Display results
function displayResults(result) {
    resultSection.style.display = 'block';

    const therapyPlan = result.result?.antibiotic_therapy_plan || {};
    const firstChoice = therapyPlan.first_choice || [];
    const secondChoice = therapyPlan.second_choice || [];
    const alternative = therapyPlan.alternative_antibiotic || [];
    const resistanceGenes = result.result?.pharmacist_analysis_on_resistant_gene || [];

    let html = '<div class="result-summary">';
    html += `<div class="summary-card"><h3>First Choice</h3><div class="count">${firstChoice.length}</div></div>`;
    html += `<div class="summary-card"><h3>Second Choice</h3><div class="count">${secondChoice.length}</div></div>`;
    html += `<div class="summary-card"><h3>Alternative</h3><div class="count">${alternative.length}</div></div>`;
    html += `<div class="summary-card"><h3>Resistance Genes</h3><div class="count">${resistanceGenes.length}</div></div>`;
    html += '</div>';

    // Display antibiotics
    if (firstChoice.length > 0) {
        html += '<h3>First Choice Antibiotics</h3>';
        html += '<ul style="text-align: left !important; margin-left: 0 !important; padding-left: 0 !important; width: 100% !important;">';
        firstChoice.forEach((ab, idx) => {
            const firstLineBadge = idx === 0 ? '<span class="first-line-badge" style="display: inline-block !important; margin-left: 0 !important; margin-right: 12px !important; text-align: center !important;">First Line</span>' : '';
            html += `<li style="text-align: left !important; margin: 8px 0 !important; margin-left: 0 !important; padding: 32px 24px !important; min-height: 100px !important; display: block !important; width: 100% !important; box-sizing: border-box !important;">
                <div class="antibiotic-header" style="text-align: left !important; margin-left: 0 !important; margin-bottom: 8px !important; display: block !important; width: 100% !important;">
                    ${firstLineBadge}
                    <strong class="antibiotic-name" style="text-align: left !important; margin-left: 0 !important; display: inline-block !important; vertical-align: middle !important;">${ab.medical_name}</strong>
                </div>
                <small style="text-align: left !important; margin-left: 0 !important; margin-top: 8px !important; display: block !important; width: 100% !important;">${ab.coverage_for || 'N/A'} | ${ab.route_of_administration || 'N/A'}</small>
            </li>`;
        });
        html += '</ul>';
    }

    if (secondChoice.length > 0) {
        html += '<h3>Second Choice Antibiotics</h3>';
        html += '<ul style="text-align: left !important; margin-left: 0 !important; padding-left: 0 !important; width: 100% !important;">';
        secondChoice.forEach(ab => {
            html += `<li style="text-align: left !important; margin: 8px 0 !important; margin-left: 0 !important; padding: 32px 24px !important; min-height: 100px !important; display: block !important; width: 100% !important; box-sizing: border-box !important;">
                <strong style="text-align: left !important; margin-left: 0 !important; display: block !important; margin-bottom: 4px !important;">${ab.medical_name}</strong>
                <small style="text-align: left !important; margin-left: 0 !important; display: block !important; width: 100% !important;">${ab.coverage_for || 'N/A'} | ${ab.route_of_administration || 'N/A'}</small>
            </li>`;
        });
        html += '</ul>';
    }

    if (alternative.length > 0) {
        html += '<h3>Alternative Antibiotics</h3>';
        html += '<ul style="text-align: left !important; margin-left: 0 !important; padding-left: 0 !important; width: 100% !important;">';
        alternative.forEach(ab => {
            html += `<li style="text-align: left !important; margin: 8px 0 !important; margin-left: 0 !important; padding: 32px 24px !important; min-height: 100px !important; display: block !important; width: 100% !important; box-sizing: border-box !important;">
                <strong style="text-align: left !important; margin-left: 0 !important; display: block !important; margin-bottom: 4px !important;">${ab.medical_name}</strong>
                <small style="text-align: left !important; margin-left: 0 !important; display: block !important; width: 100% !important;">${ab.coverage_for || 'N/A'} | ${ab.route_of_administration || 'N/A'}</small>
            </li>`;
        });
        html += '</ul>';
    }

    resultContent.innerHTML = html;
}

// Download JSON
downloadJsonBtn.addEventListener('click', () => {
    if (!currentResult) return;

    const blob = new Blob([JSON.stringify(currentResult, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `pipeline_result_${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
});

// Download PDF
downloadPdfBtn.addEventListener('click', async () => {
    if (!currentResult) {
        alert('No result available to download');
        return;
    }

    try {
        downloadPdfBtn.disabled = true;
        downloadPdfBtn.textContent = 'Generating PDF...';

        // Prepare data for PDF generation
        const pdfData = {
            data: currentResult
        };

        // If we have a session_id, use it
        if (currentSessionId) {
            pdfData.session_id = currentSessionId;
        }

        const response = await fetch(`${API_BASE_URL}/api/download-pdf`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(pdfData)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to generate PDF');
        }

        // Get PDF blob
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        
        // Get filename from Content-Disposition header or use default
        const contentDisposition = response.headers.get('Content-Disposition');
        let filename = `pipeline_report_${new Date().toISOString().split('T')[0]}.pdf`;
        if (contentDisposition) {
            const filenameMatch = contentDisposition.match(/filename="(.+)"/);
            if (filenameMatch) {
                filename = filenameMatch[1];
            }
        }
        
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        downloadPdfBtn.disabled = false;
        downloadPdfBtn.textContent = 'Download PDF';
    } catch (error) {
        console.error('Error downloading PDF:', error);
        alert(`Error generating PDF: ${error.message}`);
        downloadPdfBtn.disabled = false;
        downloadPdfBtn.textContent = 'Download PDF';
    }
});

// View raw JSON
viewJsonBtn.addEventListener('click', () => {
    if (!currentResult) return;

    const modal = document.createElement('div');
    modal.className = 'modal';
    modal.style.display = 'block';
    modal.innerHTML = `
        <div class="modal-content">
            <span class="close">&times;</span>
            <h2>Raw JSON Result</h2>
            <div class="json-viewer">${JSON.stringify(currentResult, null, 2)}</div>
        </div>
    `;
    document.body.appendChild(modal);

    modal.querySelector('.close').onclick = () => {
        modal.style.display = 'none';
        document.body.removeChild(modal);
    };

    modal.onclick = (e) => {
        if (e.target === modal) {
            modal.style.display = 'none';
            document.body.removeChild(modal);
        }
    };
});

// Check for active session on page load and automatically reconnect
async function checkActiveSession() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/sessions/active`);
        const data = await response.json();
        
        if (data.active && data.session_id) {
            // There's an active session, automatically reconnect to it
            currentSessionId = data.session_id;
            const session = data.session;
            
            // Show progress section immediately
            progressSection.style.display = 'block';
            resultSection.style.display = 'none';
            progressLog.innerHTML = '';
            
            // Update UI with current state from database
            const progress = session.progress || 0;
            const stage = session.current_stage || 'running';
            const message = session.current_stage ? `Continuing: ${stage}` : 'Pipeline running...';
            
            updateProgress(progress, message);
            addLog(`Automatically reconnected to session ${currentSessionId.substring(0, 8)}...`, 'success');
            if (session.current_stage) {
                addLog(`Current stage: ${session.current_stage}`, 'info');
            }
            
            // Disable run button
            runPipelineBtn.disabled = true;
            runPipelineBtn.textContent = 'Pipeline Running...';
            
            // Automatically connect to progress stream
            connectToProgressStream(currentSessionId);
            
            console.log('Automatically reconnected to running session:', currentSessionId);
        } else {
            console.log('No active session found');
        }
    } catch (error) {
        console.error('Error checking active session:', error);
    }
}

// Load past executions
async function loadSessions() {
    try {
        sessionsTableBody.innerHTML = '<tr><td colspan="6" class="loading">Loading sessions...</td></tr>';
        
        const response = await fetch(`${API_BASE_URL}/api/sessions?limit=50`);
        const data = await response.json();
        const sessions = data.sessions || [];
        
        if (sessions.length === 0) {
            sessionsTableBody.innerHTML = '<tr><td colspan="6" class="loading">No sessions found</td></tr>';
            return;
        }
        
        sessionsTableBody.innerHTML = '';
        
        sessions.forEach(session => {
            const row = document.createElement('tr');
            
            const sessionId = session.session_id || '';
            const status = session.status || 'unknown';
            const progress = session.progress || 0;
            const stage = session.current_stage || '-';
            const createdAt = session.created_at ? new Date(session.created_at).toLocaleString() : '-';
            
            // For running sessions, show "Viewing" if it's the current session, otherwise show "View"
            // For completed sessions, show both "View" and "Download PDF" buttons
            let actionButton = '';
            if (status === 'running') {
                if (currentSessionId === sessionId) {
                    actionButton = '<span style="color: #6b7280; font-size: 11px;">Viewing...</span>';
                } else {
                    actionButton = `<button class="btn btn-small btn-secondary" onclick="resumeSession('${sessionId}')">View</button>`;
                }
            } else if (status === 'completed') {
                actionButton = `
                    <button class="btn btn-small btn-secondary" onclick="viewSession('${sessionId}')" style="margin-right: 4px;">View</button>
                    <button class="btn btn-small btn-primary" onclick="downloadPdfFromSession('${sessionId}')">PDF</button>
                `;
            } else {
                actionButton = `<button class="btn btn-small btn-secondary" onclick="viewSession('${sessionId}')">View</button>`;
            }
            
            row.innerHTML = `
                <td class="session-id" title="${sessionId}">${sessionId.substring(0, 8)}...</td>
                <td><span class="status ${status}">${status}</span></td>
                <td class="progress">${progress}%</td>
                <td class="stage">${stage}</td>
                <td class="timestamp">${createdAt}</td>
                <td class="actions">${actionButton}</td>
            `;
            
            sessionsTableBody.appendChild(row);
        });
    } catch (error) {
        console.error('Error loading sessions:', error);
        sessionsTableBody.innerHTML = '<tr><td colspan="6" class="loading">Error loading sessions</td></tr>';
    }
}

// Switch to/view a running session (global function for onclick)
// Note: Auto-resume happens on page load, this is for manually switching to another running session
window.resumeSession = async function(sessionId) {
    try {
        // First check if session is still running
        const response = await fetch(`${API_BASE_URL}/api/sessions/${sessionId}`);
        const data = await response.json();
        
        if (data.status === 'completed' && data.result) {
            // Session completed, show results
            currentResult = data.result;
            displayResults(data.result);
            resultSection.style.display = 'block';
            progressSection.style.display = 'none';
            updateProgress(100, 'Pipeline completed');
            runPipelineBtn.disabled = false;
            runPipelineBtn.textContent = 'Run Pipeline';
            loadSessions();
            return;
        }
        
        if (data.status === 'error') {
            alert(`Session ended with error: ${data.error_message || 'Unknown error'}`);
            loadSessions();
            return;
        }
        
        // Close existing event source if any
        if (eventSource) {
            eventSource.close();
        }
        
        currentSessionId = sessionId;
        
        // Show progress section
        progressSection.style.display = 'block';
        resultSection.style.display = 'none';
        progressLog.innerHTML = '';
        
        // Update UI with current state
        const progress = data.progress || 0;
        const stage = data.current_stage || 'running';
        updateProgress(progress, `Viewing: ${stage}`);
        addLog(`Switched to session ${sessionId.substring(0, 8)}...`, 'success');
        
        // Disable run button
        runPipelineBtn.disabled = true;
        runPipelineBtn.textContent = 'Pipeline Running...';
        
        // Connect to progress stream
        connectToProgressStream(sessionId);
        
        // Refresh sessions table to update "Viewing..." status
        loadSessions();
    } catch (error) {
        console.error('Error switching to session:', error);
        alert('Error switching to session');
    }
};

// View a completed session (global function for onclick)
window.viewSession = async function(sessionId) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/sessions/${sessionId}`);
        const data = await response.json();
        
        if (data.result) {
            currentResult = data.result;
            displayResults(data.result);
            resultSection.style.display = 'block';
            progressSection.style.display = 'none';
        } else {
            alert('No result available for this session');
        }
    } catch (error) {
        console.error('Error viewing session:', error);
        alert('Error loading session data');
    }
};

// Download PDF from a session (global function for onclick)
window.downloadPdfFromSession = async function(sessionId) {
    try {
        // Use the GET endpoint for direct download
        const response = await fetch(`${API_BASE_URL}/api/download-pdf/${sessionId}`);
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to generate PDF');
        }
        
        // Get PDF blob
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        
        // Get filename from Content-Disposition header or use default
        const contentDisposition = response.headers.get('Content-Disposition');
        let filename = `pipeline_report_${sessionId.substring(0, 8)}.pdf`;
        if (contentDisposition) {
            const filenameMatch = contentDisposition.match(/filename="(.+)"/);
            if (filenameMatch) {
                filename = filenameMatch[1];
            }
        }
        
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    } catch (error) {
        console.error('Error downloading PDF:', error);
        alert(`Error generating PDF: ${error.message}`);
    }
};

// Refresh sessions list
refreshSessionsBtn.addEventListener('click', () => {
    loadSessions();
});

// Initialize on page load
document.addEventListener('DOMContentLoaded', async () => {
    // First check for active session and auto-resume (this is automatic, no manual action needed)
    await checkActiveSession();
    // Then load sessions table
    loadSessions();

});

