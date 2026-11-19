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
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class Test3 {

    public static void main(String[] args) {
        ChatModel modele = GoogleGenAiChatModel.builder()
                .apiKey(System.getenv("GEMINI_KEY"))
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

        ContentRetriever recuperateur = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(magasin)
                .embeddingModel(modeleEmbedding)
                .maxResults(2)
                .minScore(0.5)
                .build();

        System.out.println(
                "rag.pdf : " + segments.size() + " passages enregistrés."
        );

        PromptTemplate template = PromptTemplate.from("""
                Est-ce que la requête suivante porte sur l'intelligence artificielle ?
                Réponds seulement par 'oui', 'non' ou 'peut-être'.

                Requête : {{question}}
                """);

        QueryRouter routeur = new QueryRouter() {
            @Override
            public Collection<ContentRetriever> route(Query query) {
                String prompt = template.apply(
                        Map.of("question", query.text())
                ).text();

                String decision = modele.chat(prompt).trim();

                System.out.println("Décision du routeur : " + decision);

                if ("non".equalsIgnoreCase(decision)) {
                    return List.of();
                }

                return List.of(recuperateur);
            }
        };

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
}