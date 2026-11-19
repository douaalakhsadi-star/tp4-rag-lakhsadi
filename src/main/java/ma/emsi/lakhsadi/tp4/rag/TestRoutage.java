package ma.emsi.lakhsadi.tp4.rag;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.bgesmallenv15.BgeSmallEnV15EmbeddingModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class TestRoutage {

    public static void main(String[] args) {
        ChatModel modele = GoogleGenAiChatModel.builder()
                .apiKey(System.getenv("GEMINI_KEY"))
                .modelName("gemini-3.5-flash-lite")
                .temperature(0.7)
                .logRequestsAndResponses(true)
                .build();

        EmbeddingModel modeleEmbedding = new BgeSmallEnV15EmbeddingModel();

        EmbeddingStore<TextSegment> magasinRag =
                creerMagasin("rag.pdf", modeleEmbedding);

        EmbeddingStore<TextSegment> magasinIot =
                creerMagasin("atelier_6_iot_email.pdf", modeleEmbedding);

        ContentRetriever recuperateurRag =
                creerRecuperateur(magasinRag, modeleEmbedding);

        ContentRetriever recuperateurIot =
                creerRecuperateur(magasinIot, modeleEmbedding);

        Map<ContentRetriever, String> sources = new LinkedHashMap<>();

        sources.put(
                recuperateurRag,
                "Support de cours sur le fine-tuning et le RAG : "
                        + "modèles de langage, embeddings, bases vectorielles, "
                        + "recherche sémantique et LangChain4j."
        );

        sources.put(
                recuperateurIot,
                "Atelier sur un pipeline IoT en temps réel : "
                        + "ESP32, capteur DHT22, Wokwi, MQTT, HiveMQ, "
                        + "Node-RED, InfluxDB Cloud, Grafana Cloud "
                        + "et alertes par email."
        );

        QueryRouter routeur = new LanguageModelQueryRouter(modele, sources);

        RetrievalAugmentor augmentateur = DefaultRetrievalAugmentor.builder()
                .queryRouter(routeur)
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(modele)
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

    private static EmbeddingStore<TextSegment> creerMagasin(
            String nomFichier,
            EmbeddingModel modeleEmbedding) {

        Document document = ClassPathDocumentLoader.loadDocument(
                nomFichier,
                new ApacheTikaDocumentParser()
        );

        List<TextSegment> segments = DocumentSplitters
                .recursive(1_000, 100)
                .split(document);

        List<Embedding> embeddings = modeleEmbedding
                .embedAll(segments)
                .content();

        EmbeddingStore<TextSegment> magasin = new InMemoryEmbeddingStore<>();
        magasin.addAll(embeddings, segments);

        System.out.println(
                nomFichier + " : " + segments.size()
                        + " passages enregistrés."
        );

        return magasin;
    }

    private static ContentRetriever creerRecuperateur(
            EmbeddingStore<TextSegment> magasin,
            EmbeddingModel modeleEmbedding) {

        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(magasin)
                .embeddingModel(modeleEmbedding)
                .maxResults(2)
                .minScore(0.5)
                .build();
    }
}