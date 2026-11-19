package ma.emsi.lakhsadi.tp4.rag;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.bgesmallenv15.BgeSmallEnV15EmbeddingModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.Content;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.transformer.CompressingQueryTransformer;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.List;
import java.util.Scanner;

public class RagNaif {

    public static void main(String[] args) {
        String cle = System.getenv("GEMINI_KEY");

        ChatModel modele = GoogleGenAiChatModel.builder()
                .apiKey(cle)
                .modelName("gemini-3.5-flash-lite")
                .temperature(0.7)
                .logRequestsAndResponses(true)
                .build();

        Document document = ClassPathDocumentLoader.loadDocument(
                "rag.pdf",
                new ApacheTikaDocumentParser()
        );

        System.out.println(
                "Caractères extraits du PDF : " + document.text().length()
        );

        DocumentSplitter splitter = DocumentSplitters.recursive(1_000, 100);
        List<TextSegment> segments = splitter.split(document);

        System.out.println("Nombre de passages : " + segments.size());

        EmbeddingModel modeleEmbedding = new BgeSmallEnV15EmbeddingModel();

        List<Embedding> embeddings = modeleEmbedding
                .embedAll(segments)
                .content();

        System.out.println("Nombre d'embeddings : " + embeddings.size());

        EmbeddingStore<TextSegment> magasin = new InMemoryEmbeddingStore<>();
        magasin.addAll(embeddings, segments);

        System.out.println("Passages et embeddings enregistrés en mémoire.");
        System.out.println("=== Passages enregistrés ===");

        for (int i = 0; i < segments.size(); i++) {
            System.out.println("Passage " + (i + 1));
            System.out.println(segments.get(i).text());
            System.out.println();
        }

        ContentRetriever recuperateur = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(magasin)
                .embeddingModel(modeleEmbedding)
                .maxResults(2)
                .minScore(0.5)
                .build();

        ChatMemory memoire = MessageWindowChatMemory.withMaxMessages(10);

        // Reformuler la question en utilisant l'historique.
        CompressingQueryTransformer transformateur =
                new CompressingQueryTransformer(modele);

        RetrievalAugmentor augmentateur = DefaultRetrievalAugmentor.builder()
                .queryTransformer(transformateur)
                .contentRetriever(recuperateur)
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(modele)
                .chatMemory(memoire)
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

                // Diagnostic sur la question originale, sans transformation.
                List<Content> contenus = recuperateur.retrieve(
                        new Query(question)
                );

                System.out.println(
                        "=== Diagnostic : passages pour la question originale ==="
                );

                for (int i = 0; i < contenus.size(); i++) {
                    System.out.println("Résultat " + (i + 1));
                    System.out.println(contenus.get(i).textSegment().text());
                    System.out.println();
                }

                if (contenus.isEmpty()) {
                    System.out.println(
                            "Aucun passage ne satisfait le seuil de similarité."
                    );
                }

                Embedding embeddingQuestion = modeleEmbedding
                        .embed(question)
                        .content();

                EmbeddingSearchRequest recherche = EmbeddingSearchRequest.builder()
                        .queryEmbedding(embeddingQuestion)
                        .maxResults(2)
                        .minScore(0.5)
                        .build();

                List<EmbeddingMatch<TextSegment>> resultats = magasin
                        .search(recherche)
                        .matches();

                System.out.println(
                        "=== Diagnostic : scores pour la question originale ==="
                );

                for (EmbeddingMatch<TextSegment> resultat : resultats) {
                    System.out.printf("Score : %.4f%n", resultat.score());
                    System.out.println(resultat.embedded().text());
                    System.out.println();
                }

                if (resultats.isEmpty()) {
                    System.out.println(
                            "Aucun passage avec un score supérieur ou égal à 0,5."
                    );
                }

                // L'assistant utilise le transformateur avant la récupération.
                String reponse = assistant.chat(question);
                System.out.println("Réponse : " + reponse);
                System.out.println();
            }
        }
    }
}