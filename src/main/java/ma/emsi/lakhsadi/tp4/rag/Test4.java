package ma.emsi.lakhsadi.tp4.rag;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.bgesmallenv15.BgeSmallEnV15EmbeddingModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class Test4 {

    public static void main(String[] args) {
        String cleGemini = System.getenv("GEMINI_KEY");
        String cleTavily = System.getenv("TAVILY_KEY");

        if (cleGemini == null || cleGemini.isBlank()) {
            throw new IllegalStateException("La variable GEMINI_KEY est absente.");
        }

        if (cleTavily == null || cleTavily.isBlank()) {
            throw new IllegalStateException("La variable TAVILY_KEY est absente.");
        }

        ChatModel modele = GoogleGenAiChatModel.builder()
                .apiKey(cleGemini)
                .modelName("gemini-3.5-flash-lite")
                .temperature(0.7)
                .logRequestsAndResponses(true)
                .build();

        Document document = ClassPathDocumentLoader.loadDocument(
                "rag.pdf",
                new ApacheTikaDocumentParser()
        );

        List<TextSegment> segments = DocumentSplitters
                .recursive(1_000, 100)
                .split(document);

        EmbeddingModel modeleEmbedding = new BgeSmallEnV15EmbeddingModel();

        List<Embedding> embeddings = modeleEmbedding
                .embedAll(segments)
                .content();

        EmbeddingStore<TextSegment> magasin = new InMemoryEmbeddingStore<>();
        magasin.addAll(embeddings, segments);

        System.out.println(
                "rag.pdf : " + segments.size() + " passages enregistrés."
        );

        ContentRetriever recuperateurPdf =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(magasin)
                        .embeddingModel(modeleEmbedding)
                        .maxResults(2)
                        .minScore(0.5)
                        .build();

        WebSearchEngine moteurWeb = TavilyWebSearchEngine.builder()
                .apiKey(cleTavily)
                .build();

        ContentRetriever recuperateurWeb = WebSearchContentRetriever.builder()
                .webSearchEngine(moteurWeb)
                .build();

        // Ancien routage : les deux sources étaient toujours consultées.
        /*
        QueryRouter routeur =
                new dev.langchain4j.rag.query.router.DefaultQueryRouter(
                        recuperateurPdf,
                        recuperateurWeb
                );
        */

        // Nouveau routage : le modèle choisit les sources adaptées.
        Map<ContentRetriever, String> sources = new LinkedHashMap<>();

        sources.put(
                recuperateurPdf,
                "Support de cours sur le RAG, le fine-tuning, les embeddings, "
                        + "les bases vectorielles et LangChain4j. "
                        + "À utiliser pour les questions sur ces notions "
                        + "ou sur ce support de cours. "
                        + "Ne contient pas de résultats sportifs "
                        + "ni d'actualités."
        );

        sources.put(
                recuperateurWeb,
                "Recherche sur le Web pour les actualités, "
                        + "les événements récents, les résultats sportifs "
                        + "et les sujets extérieurs au support de cours. "
                        + "À utiliser aussi pour les informations récentes "
                        + "sur l'IA absentes du support."
        );

        QueryRouter routeur = new LanguageModelQueryRouter(modele, sources);

        RetrievalAugmentor augmentateur = DefaultRetrievalAugmentor.builder()
                .queryRouter(routeur)
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(modele)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(augmentateur)
                .build();

        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.println(
                        "Posez votre question, ou tapez fin pour quitter :"
                );

                if (!scanner.hasNextLine()) {
                    break;
                }

                String question = scanner.nextLine().trim();

                if (question.equalsIgnoreCase("fin")) {
                    break;
                }

                if (question.isBlank()) {
                    continue;
                }

                String reponse = assistant.chat(question);
                System.out.println("Réponse : " + reponse);
                System.out.println();
            }
        }
    }
}