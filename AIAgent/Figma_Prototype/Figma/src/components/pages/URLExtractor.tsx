import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Page } from "../../App";
import { Loader2, Globe, FileText, CheckCircle, AlertCircle, ArrowLeft } from "lucide-react";
import { motion } from "motion/react";
import { Alert, AlertDescription } from "../ui/alert";

interface URLExtractorProps {
  navigateTo: (page: Page) => void;
}

interface ExtractionResult {
  url: string;
  raw_html: string;
  parsed_text: string;
  cleaned_text: string;
  success: boolean;
}

const API_BASE_URL = "http://localhost:8000";

export function URLExtractor({ navigateTo }: URLExtractorProps) {
  const [url, setUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ExtractionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleExtract = async () => {
    if (!url.trim()) {
      setError("Please enter a valid URL");
      return;
    }

    try {
      new URL(url);
    } catch {
      setError("Please enter a valid URL (e.g., https://example.com)");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/run-flow`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ url: url.trim() }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to extract content");
      }

      const data = await response.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || "Failed to connect to backend. Make sure the server is running at http://localhost:8000");
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !loading) {
      handleExtract();
    }
  };

  return (
    <div className="min-h-screen p-6">
      <div className="max-w-5xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <div className="flex items-center gap-4 mb-6">
            <Button
              variant="outline"
              size="sm"
              onClick={() => navigateTo("homepage")}
            >
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back
            </Button>
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
                URL Content Extractor
              </h1>
              <p className="text-muted-foreground mt-2">
                Extract clean text from any website using MCP agents
              </p>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="mb-6"
        >
          <Card className="bg-white/70 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Globe className="h-5 w-5 text-indigo-600" />
                Enter Website URL
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex gap-2">
                <Input
                  type="url"
                  placeholder="https://example.com"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  onKeyPress={handleKeyPress}
                  disabled={loading}
                  className="flex-1"
                />
                <Button
                  onClick={handleExtract}
                  disabled={loading || !url.trim()}
                  className="bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700"
                >
                  {loading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <FileText className="mr-2 h-4 w-4" />
                      Extract
                    </>
                  )}
                </Button>
              </div>

              <div className="text-sm text-muted-foreground">
                <p>✨ Using MCP Agents:</p>
                <ul className="list-disc list-inside ml-4 mt-1 space-y-1">
                  <li>Crawler Agent - Fetches web content</li>
                  <li>Parser Agent - Extracts clean text</li>
                </ul>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-6"
          >
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          </motion.div>
        )}

        {result && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="space-y-6"
          >
            {result.success && (
              <Alert className="bg-green-50 border-green-200">
                <CheckCircle className="h-4 w-4 text-green-600" />
                <AlertDescription className="text-green-800">
                  Successfully extracted content from {result.url}
                </AlertDescription>
              </Alert>
            )}

            <Card className="bg-white/70 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <FileText className="h-5 w-5 text-green-600" />
                  Extracted Text
                </CardTitle>
                <p className="text-sm text-muted-foreground">
                  Clean, readable text extracted from the website
                </p>
              </CardHeader>
              <CardContent>
                <div className="bg-slate-50 rounded-lg p-4 max-h-96 overflow-y-auto">
                  <p className="whitespace-pre-wrap text-sm leading-relaxed">
                    {result.cleaned_text || result.parsed_text || "No text extracted"}
                  </p>
                </div>
                <div className="mt-4 text-xs text-muted-foreground">
                  <p>Character count: {result.cleaned_text?.length || 0}</p>
                </div>
              </CardContent>
            </Card>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Card className="bg-white/70 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="text-sm">Raw HTML Length</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-2xl font-bold text-indigo-600">
                    {result.raw_html?.length || 0}
                  </p>
                  <p className="text-xs text-muted-foreground">characters</p>
                </CardContent>
              </Card>

              <Card className="bg-white/70 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="text-sm">Parsed Text Length</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-2xl font-bold text-purple-600">
                    {result.parsed_text?.length || 0}
                  </p>
                  <p className="text-xs text-muted-foreground">characters</p>
                </CardContent>
              </Card>
            </div>
          </motion.div>
        )}

        {!result && !loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
            className="mt-8"
          >
            <Card className="bg-blue-50 border-blue-200">
              <CardHeader>
                <CardTitle className="text-sm">How it works</CardTitle>
              </CardHeader>
              <CardContent className="text-sm space-y-2">
                <p>1. Enter any website URL above</p>
                <p>2. Click "Extract" to start the agent workflow</p>
                <p>3. The Crawler Agent fetches the webpage</p>
                <p>4. The Parser Agent extracts clean text</p>
                <p>5. View the extracted content below</p>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </div>
    </div>
  );
}

