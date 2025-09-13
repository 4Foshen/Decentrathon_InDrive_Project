import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'result_page.dart';
import 'package:indrive_ai/theme/app_colors.dart';

class CarPhotoPage extends StatefulWidget {
  final String driverName;
  final String carModel;

  const CarPhotoPage({
    super.key,
    required this.driverName,
    required this.carModel,
  });

  @override
  State<CarPhotoPage> createState() => _CarPhotoPageState();
}

class _CarPhotoPageState extends State<CarPhotoPage> {
  File? carPhoto; // фото сохраняем как File
  final ImagePicker _picker = ImagePicker();

  Future<void> _uploadPhoto() async {
    final XFile? pickedFile = await _picker.pickImage(
      source: ImageSource.camera, // можно заменить на ImageSource.gallery
      imageQuality: 80,
    );

    if (pickedFile != null) {
      setState(() {
        carPhoto = File(pickedFile.path);
      });
    }
  }

  void _goNext() {
    if (carPhoto == null) return;

    Navigator.push(
      context,
      buildPageTransition(
        ResultPage(
          driverName: widget.driverName,
          carModel: widget.carModel,
          carPhoto: carPhoto!.path, // передаём путь как строку
        ),
        type: TransitionType.scale,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        title: const Text("Фото машины"),
        backgroundColor: Colors.transparent,
        elevation: 0,
        foregroundColor: AppColors.textPrimary,
      ),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // Фото или иконка
            carPhoto != null
                ? ClipRRect(
                    borderRadius: BorderRadius.circular(16),
                    child: Image.file(
                      carPhoto!,
                      height: 300,
                      width: 300,
                      fit: BoxFit.cover,
                    ),
                  )
                : Column(
                    children: [
                      GestureDetector(
                        onTap: _uploadPhoto,
                        child: Container(
                          padding: const EdgeInsets.all(24),
                          decoration: BoxDecoration(
                            color: AppColors.surface,
                            shape: BoxShape.circle,
                            boxShadow: [
                              BoxShadow(
                                color: Colors.black.withOpacity(0.1),
                                blurRadius: 8,
                                offset: const Offset(0, 4),
                              ),
                            ],
                          ),
                          child: const Icon(
                            Icons.camera_alt,
                            size: 60,
                            color: AppColors.primary,
                          ),
                        ),
                      ),
                      const SizedBox(height: 12),
                      const Text(
                        "Загрузите фотографию машины",
                        style: TextStyle(
                          fontSize: 16,
                          color: AppColors.textSecondary,
                        ),
                      ),
                    ],
                  ),
            const SizedBox(height: 40),

            // Кнопка Далее
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: _goNext,
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppColors.primary,
                  padding:
                      const EdgeInsets.symmetric(vertical: 14, horizontal: 30),
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12)),
                  elevation: 3,
                ),
                child: const Text(
                  "Далее",
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                    color: AppColors.textPrimary,
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Универсальные транзишены
Route buildPageTransition(Widget page,
    {TransitionType type = TransitionType.slide}) {
  return PageRouteBuilder(
    transitionDuration: const Duration(milliseconds: 600),
    pageBuilder: (_, __, ___) => page,
    transitionsBuilder: (_, animation, __, child) {
      switch (type) {
        case TransitionType.slide:
          return SlideTransition(
            position: Tween(begin: const Offset(1, 0), end: Offset.zero)
                .animate(CurvedAnimation(
                    parent: animation, curve: Curves.easeOutCubic)),
            child: FadeTransition(opacity: animation, child: child),
          );
        case TransitionType.scale:
          return ScaleTransition(
            scale: Tween(begin: 0.9, end: 1.0).animate(
                CurvedAnimation(parent: animation, curve: Curves.easeOutBack)),
            child: FadeTransition(opacity: animation, child: child),
          );
        default:
          return child;
      }
    },
  );
}

enum TransitionType { slide, scale }
