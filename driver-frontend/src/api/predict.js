// driver-frontend/src/api/predict.js
export async function predictRoutes({ from, to, departTime }) {
  //const API_URL = process.env.REACT_APP_ML_API_URL || 'http://localhost:5000';
  // const API_URL = process.env.REACT_APP_ML_API_URL;
  const API_URL = '';

  try {
    console.log('🔍 Calling ML API:', API_URL);
    console.log('📍 From:', from, 'To:', to);

    // const response = await fetch(`${API_URL}/predict`, {
    const response = await fetch(`/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        from,
        to,
        departTime: departTime || new Date().toISOString(),
      }),
    });

    console.log('📡 Response status:', response.status);

    if (!response.ok) {
      const errorData = await response.json();
      console.error('❌ API Error:', errorData);
      throw new Error(errorData.error || 'Prediction failed');
    }

    const data = await response.json();
    console.log('✅ ML Response:', data);

    // Backend now returns { best, alternatives, note, explanation }
    // Return as-is for the dialog
    return {
      best: data.best,
      alternatives: data.alternatives || [],
      note: data.note,
      explanation: data.explanation
    };

  } catch (error) {
    console.error('🔥 Prediction error:', error);

    // Fallback response
    return {
      best: {
        id: 'fallback',
        route_name: `${from} → ${to}`,
        label: '⚠️ API Offline',
        duration_min: 25,
        distance_km: 12,
        confidence: 0.7,
        congestionProb: 0.35,
        status: 'clear'
      },
      alternatives: [],
      note: `Error: ${error.message}`,
    };
  }
}