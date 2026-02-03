export default async function handler(req, res) {
  // Set CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  
  // Handle preflight
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { frames } = req.body;

    if (!frames || !Array.isArray(frames) || frames.length !== 4) {
      return res.status(400).json({ error: 'Invalid frames data. Expected 4 frames.' });
    }

    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) {
      return res.status(500).json({ error: 'API key not configured' });
    }

    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01'
      },
      body: JSON.stringify({
        model: 'claude-sonnet-4-20250514',
        max_tokens: 2000,
        messages: [{
          role: 'user',
          content: [
            ...frames.map(frame => ({
              type: 'image',
              source: {
                type: 'base64',
                media_type: 'image/jpeg',
                data: frame
              }
            })),
            {
              type: 'text',
              text: 'You are an expert golf instructor. Analyze these 4 swing frames and identify faults. Return JSON: {"faults": [{"name": "...", "description": "..."}], "coaching_tips": ["...", "..."]}'
            }
          ]
        }]
      })
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      return res.status(response.status).json({ error: errorData.error?.message || 'Claude API error' });
    }

    const data = await response.json();
    return res.status(200).json(data);

  } catch (error) {
    console.error('Analysis error:', error);
    return res.status(500).json({ error: 'Analysis failed' });
  }
}

